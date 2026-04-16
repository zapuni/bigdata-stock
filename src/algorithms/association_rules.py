"""
Association Rules Mining -- Phase 2: Mining Massive Datasets

Bai toan: "Khi nhom chi bao A xuat hien dong thoi trong 1 phien giao dich,
nhom chi bao B co xu huong xuat hien cung khong?"

Pipeline:
  STAGE 1  load_and_encode_transactions : daily parquet -> list[items] per (stock, date)
  STAGE 2  run_fpgrowth_spark           : Spark MLlib FPGrowth (SON-style distributed)
           run_apriori_pandas (fallback): mlxtend Apriori (single-machine baseline)
  STAGE 3  filter_and_format_rules      : lift filter + rule_str + sort by lift

Ly thuyet SON (Savasere et al., 1995):
  Pass 1: chia data thanh p partitions, moi partition chay Apriori/FPGrowth
          voi nguong support thap (s_i = s * fraction) -> local frequent itemsets L_i
          UNION L_i = candidate set C (chac chan chua moi global frequent itemset)
  Pass 2: dem exact support cua moi candidate trong C tren toan dataset
          -> giu candidate co support >= s -> ket qua chinh xac

Spark FPGrowth = SON-style:
  Partition data -> moi partition build FP-tree -> mine local patterns ->
  aggregate global voi exact count filter (= Pass 1 + Pass 2 cua SON).

Input : lsh-similarity/daily/ (~169K rows, da aggregate EOD per stock-day)
Output: list of dict (frequent_itemsets, association_rules)
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

log = logging.getLogger("stock_assoc")


# ---------------------------------------------------------------------------
# STAGE 1: LOAD AND ENCODE TRANSACTIONS
# ---------------------------------------------------------------------------

def _encode_value(col_name: str, val, prefix_map: Dict[str, str],
                  numeric_label_map: Dict[str, Dict]) -> Optional[str]:
    """Bien doi 1 (column, value) thanh 1 item string 'prefix_LABEL'.

    Tra ve None neu val NaN/None.
    Numeric cols (vd is_high_volatility=1/0) duoc map sang label readable.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    prefix = prefix_map.get(col_name, col_name)

    # Numeric -> readable label
    if col_name in numeric_label_map:
        try:
            key = int(val)
            label = numeric_label_map[col_name].get(key)
            if label is not None:
                return f"{prefix}_{label}"
        except (TypeError, ValueError):
            pass

    val_str = str(val).strip().replace(" ", "_")
    if not val_str or val_str.lower() == "nan":
        return None
    return f"{prefix}_{val_str}"


def load_and_encode_transactions(
    daily_path: str,
    assoc_cols: List[str],
    config: Dict,
) -> pd.DataFrame:
    """Doc daily parquet, encode moi (stock, date) thanh 1 transaction.

    Args:
        daily_path : path toi lsh-similarity/daily/ (da daily aggregate)
        assoc_cols : cot trong FEATURE_GROUPS["association_items"]
        config     : ASSOC_CONFIG

    Returns:
        DataFrame [stock_symbol, trade_date, items]
        items = list of "prefix_LABEL" strings, vd: ["rsi_Oversold", "bb_LOWER_ZONE", ...]
    """
    date_col = config["date_col"]
    stock_col = config["stock_col"]
    prefix_map = config["prefix_map"]
    numeric_label_map = config["numeric_label_map"]

    log.info("STAGE 1 LOAD     | Reading daily parquet from %s", daily_path)
    log.info("STAGE 1 LOAD     | Columns (%d): %s", len(assoc_cols), assoc_cols)

    # daily/ duoc Spark ghi voi partitionBy("year", "stock_symbol") => 2 cot nay
    # nam trong duong dan thu muc Hive-style ("year=X/stock_symbol=Y/"), khong
    # nam trong file schema. Phai khai bao partitioning="hive" de Dataset API
    # nhan ra partition cols.
    dataset = ds.dataset(daily_path, format="parquet", partitioning="hive")
    existing = set(dataset.schema.names)
    missing = [c for c in assoc_cols if c not in existing]
    missing_meta = [c for c in (stock_col, date_col) if c not in existing]
    if missing_meta:
        raise ValueError(
            f"Missing required columns in {daily_path}: {missing_meta}. "
            f"Available: {sorted(existing)}"
        )
    if missing:
        raise ValueError(
            f"Missing columns in {daily_path}: {missing}. "
            f"Available: {sorted(existing)}"
        )

    read_cols = [stock_col, date_col] + list(assoc_cols)
    df = dataset.to_table(columns=read_cols).to_pandas()

    # stock_symbol doc tu partition Hive co the la dictionary/categorical
    # -> ep ve string de drop_duplicates / nunique nhat quan
    df[stock_col] = df[stock_col].astype(str)
    df = df.drop_duplicates(subset=[stock_col, date_col])
    log.info("STAGE 1 LOAD     | %s transactions, %d stocks",
             f"{len(df):,}", df[stock_col].nunique())

    # Diagnostic: log dtype + null% + sample values cho moi assoc col
    for c in assoc_cols:
        null_pct = df[c].isna().mean() * 100
        non_null = df[c].dropna()
        sample = non_null.iloc[0] if len(non_null) > 0 else "(all null)"
        unique_n = non_null.nunique()
        log.info("STAGE 1 LOAD     | %-22s dtype=%-8s nulls=%5.2f%% "
                 "unique=%d sample=%r",
                 c, str(df[c].dtype), null_pct, unique_n, sample)

    # Encode rows -> items list
    log.info("STAGE 1 LOAD     | Encoding %s rows...", f"{len(df):,}")
    t0 = time.time()
    arr_cols = {c: df[c].values for c in assoc_cols}
    items_list = []
    for i in range(len(df)):
        items = []
        for c in assoc_cols:
            item = _encode_value(c, arr_cols[c][i], prefix_map, numeric_label_map)
            if item is not None:
                items.append(item)
        items_list.append(items)

    df = df[[stock_col, date_col]].copy()
    df["items"] = items_list

    # Loai transaction co < 2 items (rule can it nhat 2 items)
    df["_size"] = df["items"].apply(len)
    n_before = len(df)
    df = df[df["_size"] >= 2].drop(columns=["_size"]).reset_index(drop=True)
    log.info("STAGE 1 LOAD     | Encoded in %.1fs | dropped %s tx with <2 items",
             time.time() - t0, f"{n_before - len(df):,}")

    sizes = df["items"].apply(len)
    vocab = set()
    for items in df["items"]:
        vocab.update(items)
    log.info("STAGE 1 LOAD     | Final: %s tx | avg items/tx=%.2f | vocabulary=%d items",
             f"{len(df):,}", sizes.mean(), len(vocab))

    # Sample encoded tx + vocabulary preview
    if len(df) > 0:
        log.info("STAGE 1 LOAD     | Sample tx [0]: stock=%s date=%s items=%s",
                 df.iloc[0][stock_col], df.iloc[0][date_col], df.iloc[0]["items"])
    log.info("STAGE 1 LOAD     | Vocabulary (sorted): %s",
             ", ".join(sorted(vocab)))

    return df


# ---------------------------------------------------------------------------
# STAGE 2A: SPARK FPGROWTH (SON-STYLE)
# ---------------------------------------------------------------------------

def run_fpgrowth_spark(
    transactions_df: pd.DataFrame,
    spark,
    min_support: float,
    min_confidence: float,
    max_itemset_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Spark MLlib FPGrowth (SON-style: local mine + global count).

    Args:
        transactions_df : pandas df voi col "items" = list[str]
        spark           : SparkSession
        min_support     : 0..1
        min_confidence  : 0..1
        max_itemset_size: gioi han len(itemset) de tranh combinatorial explosion

    Returns:
        (frequent_itemsets_df, rules_df) -- ca 2 deu pandas
    """
    from pyspark.ml.fpm import FPGrowth

    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    log.info("STAGE 2 FPGROWTH | Spark (SON-style distributed)")
    n_total = len(transactions_df)
    log.info("STAGE 2 FPGROWTH | Converting %s transactions to Spark DF...",
             f"{n_total:,}")

    # FPGrowth yeu cau items unique trong 1 transaction. Voi prefix_map,
    # moi col -> 1 item rieng biet -> da unique trong design.
    # Dung tuples + explicit schema de tranh Spark infer sai khi list rong/None.
    items_list = transactions_df["items"].tolist()
    schema = StructType([
        StructField("items", ArrayType(StringType(), containsNull=False), False),
    ])
    spark_df = spark.createDataFrame(
        [(items,) for items in items_list],
        schema=schema,
    )
    n_partitions = spark_df.rdd.getNumPartitions()
    log.info("STAGE 2 FPGROWTH | Spark partitions: %d", n_partitions)

    fpg = FPGrowth(
        itemsCol="items",
        minSupport=min_support,
        minConfidence=min_confidence,
    )
    log.info("STAGE 2 FPGROWTH | min_support=%.4f, min_confidence=%.4f",
             min_support, min_confidence)

    t0 = time.time()
    model = fpg.fit(spark_df)
    fit_time = time.time() - t0
    log.info("STAGE 2 FPGROWTH | Fit completed in %.1fs", fit_time)

    fi_df = model.freqItemsets.toPandas()
    fi_df = fi_df.rename(columns={"items": "itemset", "freq": "count"})
    fi_df["support"] = fi_df["count"] / n_total
    fi_df["itemset_size"] = fi_df["itemset"].apply(len)
    fi_df = fi_df[fi_df["itemset_size"] <= max_itemset_size]
    fi_df = fi_df.sort_values("support", ascending=False).reset_index(drop=True)

    rules_df = model.associationRules.toPandas()

    log.info("STAGE 2 FPGROWTH | Frequent itemsets: %s | Rules: %s",
             f"{len(fi_df):,}", f"{len(rules_df):,}")
    if len(fi_df) > 0:
        size_dist = fi_df["itemset_size"].value_counts().sort_index().to_dict()
        log.info("STAGE 2 FPGROWTH | Itemset size distribution: %s", size_dist)

    return fi_df, rules_df


# ---------------------------------------------------------------------------
# STAGE 2B: PANDAS APRIORI (FALLBACK / BASELINE)
# ---------------------------------------------------------------------------

def run_apriori_pandas(
    transactions_df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    max_itemset_size: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """mlxtend Apriori (single-machine baseline cho so sanh).

    Returns:
        (fi_df, rules_df) -- pandas. None neu mlxtend chua install.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        log.error("mlxtend chua cai. Cai bang: pip install mlxtend")
        return None, None

    log.info("STAGE 2 APRIORI  | mlxtend single-machine baseline")
    n_total = len(transactions_df)

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions_df["items"].tolist())
    te_df = pd.DataFrame(te_array, columns=te.columns_)
    log.info("STAGE 2 APRIORI  | Binary matrix: %s rows x %d items (%.1f MB)",
             f"{te_df.shape[0]:,}", te_df.shape[1],
             te_df.memory_usage(deep=True).sum() / 1e6)

    t0 = time.time()
    fi_df = apriori(
        te_df, min_support=min_support, max_len=max_itemset_size,
        use_colnames=True,
    )
    fit_time = time.time() - t0
    log.info("STAGE 2 APRIORI  | Apriori completed in %.1fs", fit_time)

    if len(fi_df) == 0:
        log.warning("STAGE 2 APRIORI  | No frequent itemsets found")
        return fi_df, pd.DataFrame()

    fi_df["count"] = (fi_df["support"] * n_total).round().astype(int)
    fi_df["itemset"] = fi_df["itemsets"].apply(lambda s: list(sorted(s)))
    fi_df["itemset_size"] = fi_df["itemset"].apply(len)
    fi_df = fi_df[["itemset", "count", "support", "itemset_size"]]
    fi_df = fi_df.sort_values("support", ascending=False).reset_index(drop=True)

    # association_rules can frozenset itemsets
    fi_for_rules = fi_df.copy()
    fi_for_rules["itemsets"] = fi_for_rules["itemset"].apply(frozenset)
    fi_for_rules = fi_for_rules[["support", "itemsets"]]

    rules_df = association_rules(
        fi_for_rules, metric="confidence", min_threshold=min_confidence,
    )
    if len(rules_df) > 0:
        rules_df["antecedent"] = rules_df["antecedents"].apply(
            lambda s: list(sorted(s)))
        rules_df["consequent"] = rules_df["consequents"].apply(
            lambda s: list(sorted(s)))
        rules_df = rules_df.drop(columns=["antecedents", "consequents"])

    log.info("STAGE 2 APRIORI  | Frequent itemsets: %s | Rules: %s",
             f"{len(fi_df):,}", f"{len(rules_df):,}")
    return fi_df, rules_df


# ---------------------------------------------------------------------------
# STAGE 3: COMPUTE LIFT (SAFETY) + FILTER + FORMAT
# ---------------------------------------------------------------------------

def _ensure_lift(rules_df: pd.DataFrame, fi_df: pd.DataFrame) -> pd.DataFrame:
    """Mot so version Spark MLlib khong return cot 'lift' -> tinh thu cong."""
    if "lift" in rules_df.columns and rules_df["lift"].notna().any():
        return rules_df
    log.warning("STAGE 3 LIFT     | Column 'lift' missing -> computing manually")

    sup_map = {}
    for _, row in fi_df.iterrows():
        items = row["itemset"]
        if isinstance(items, (list, tuple, np.ndarray)):
            sup_map[frozenset(items)] = float(row["support"])

    lifts = []
    for _, r in rules_df.iterrows():
        ant = frozenset(r["antecedent"])
        con = frozenset(r["consequent"])
        sup_union = sup_map.get(ant | con)
        sup_con = sup_map.get(con)
        sup_ant = sup_map.get(ant)
        if sup_union and sup_con and sup_ant and sup_con > 0 and sup_ant > 0:
            conf = sup_union / sup_ant
            lifts.append(conf / sup_con)
        else:
            lifts.append(np.nan)
    rules_df = rules_df.copy()
    rules_df["lift"] = lifts
    if "support" not in rules_df.columns:
        rules_df["support"] = [
            sup_map.get(frozenset(r["antecedent"]) | frozenset(r["consequent"]))
            for _, r in rules_df.iterrows()
        ]
    return rules_df


def filter_and_format_rules(
    rules_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    min_lift: float,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter rules theo min_lift, them rule_str + rank, sort by lift desc.

    Returns:
        (rules_filtered_df, top_rules_df)
    """
    if rules_df is None or len(rules_df) == 0:
        log.warning("STAGE 3 FILTER   | rules_df empty")
        empty = pd.DataFrame()
        return empty, empty

    rules_df = _ensure_lift(rules_df, fi_df)

    n_total = len(rules_df)
    rules = rules_df[rules_df["lift"] >= min_lift].copy()
    n_pass = len(rules)
    log.info("STAGE 3 FILTER   | %s rules pass lift>=%.2f (%.1f%% kept)",
             f"{n_pass:,}", min_lift, n_pass / max(n_total, 1) * 100)

    def fmt_items(items):
        if isinstance(items, (list, tuple, np.ndarray)):
            return " & ".join(sorted(str(i) for i in items))
        if isinstance(items, (frozenset, set)):
            return " & ".join(sorted(str(i) for i in items))
        return str(items)

    rules["antecedent_str"] = rules["antecedent"].apply(fmt_items)
    rules["consequent_str"] = rules["consequent"].apply(fmt_items)
    rules["rule_str"] = (
        "{" + rules["antecedent_str"] + "}"
        + " -> "
        + "{" + rules["consequent_str"] + "}"
    )

    for c in ("support", "confidence", "lift"):
        if c in rules.columns:
            rules[c] = rules[c].astype(float).round(4)

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    rules.insert(0, "rank", rules.index + 1)

    top = rules.head(top_n).copy()
    log.info("STAGE 3 FILTER   | Top 5 rules by lift:")
    for _, r in top.head(5).iterrows():
        log.info("                   [#%2d] %s", r["rank"], r["rule_str"])
        log.info("                          support=%.4f conf=%.4f lift=%.4f",
                 r.get("support", float("nan")),
                 r.get("confidence", float("nan")),
                 r.get("lift", float("nan")))

    return rules, top
