"""
Module 4 Runner: Association Rules (SON Algorithm via Spark FPGrowth)

Pipeline:
    1. Verify columns trong daily parquet
    2. Load + encode transactions (lsh-similarity/daily/ -> list[items])
    3. Save transactions.parquet
    4. Run Spark FPGrowth (SON-style) hoac mlxtend Apriori (fallback)
    5. Filter rules theo lift, format rule_str
    6. Export CSV/Parquet + 3 visualizations
    7. Sinh doan VN paste vao bao cao

Usage:
    conda activate stock

    # Default: Spark FPGrowth
    python src/run_association_rules.py

    # Pandas Apriori baseline (de so sanh)
    python src/run_association_rules.py --no-spark

    # Tune threshold
    python src/run_association_rules.py --min-support 0.03 --min-confidence 0.6

    # Quick test
    python src/run_association_rules.py --sample 0.1 --min-support 0.01

Dependencies:
    pyspark, pandas, pyarrow, matplotlib (da co)
    mlxtend (optional, cho --no-spark): pip install mlxtend
"""

import os
import sys
import time
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = SRC_DIR + (os.pathsep + _existing if _existing else "")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config.settings import (
    LSH_PATH, ASSOC_CONFIG, ASSOC_OUTPUT_PATH, ASSOC_REPORTS_DIR,
    LOGS_DIR, FEATURE_GROUPS, SPARK_CONFIG,
)
from algorithms.association_rules import (
    load_and_encode_transactions,
    run_fpgrowth_spark,
    run_apriori_pandas,
    filter_and_format_rules,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ASSOC_OUTPUT_PATH, exist_ok=True)
os.makedirs(ASSOC_REPORTS_DIR, exist_ok=True)

log = logging.getLogger("stock_assoc")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "association_rules.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


# ---------------------------------------------------------------------------
# SPARK SESSION
# ---------------------------------------------------------------------------

def _create_spark(config):
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(config.get("spark_app_name", "StockAssocRules"))
        .master(SPARK_CONFIG.get("master", "local[*]"))
        .config("spark.driver.memory",
                config.get("spark_driver_memory", SPARK_CONFIG["driver_memory"]))
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
        .config("spark.driver.maxResultSize", SPARK_CONFIG["max_result_size"])
        .config("spark.sql.adaptive.enabled", SPARK_CONFIG["adaptive_enabled"])
        .config("spark.sql.adaptive.coalescePartitions.enabled",
                SPARK_CONFIG["coalesce_partitions_enabled"])
        .config("spark.memory.fraction", SPARK_CONFIG["spark.memory.fraction"])
        .config("spark.memory.storageFraction",
                SPARK_CONFIG["spark.memory.storageFraction"])
        .config("spark.sql.shuffle.partitions",
                SPARK_CONFIG["spark.sql.shuffle.partitions"])
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def _serialize_listcol(df, col_name):
    """Bien frozenset/set/np.ndarray -> list de Parquet luu duoc."""
    if col_name not in df.columns:
        return df
    df = df.copy()
    df[col_name] = df[col_name].apply(
        lambda x: list(x) if isinstance(x, (frozenset, set, tuple, np.ndarray))
        else (x if isinstance(x, list) else [x])
    )
    return df


def export_csvs(fi_df, rules_filtered, top_rules):
    fi_path = os.path.join(ASSOC_REPORTS_DIR, "frequent_itemsets.csv")
    fi_export = fi_df.copy()
    if "itemset" in fi_export.columns:
        fi_export["itemset"] = fi_export["itemset"].apply(
            lambda x: ",".join(sorted(str(i) for i in x)))
    fi_export.to_csv(fi_path, index=False)
    log.info("EXPORT CSV       | %s (%s rows)", fi_path, f"{len(fi_export):,}")

    rules_path = os.path.join(ASSOC_REPORTS_DIR, "association_rules.csv")
    rules_export = rules_filtered.copy()
    for c in ("antecedent", "consequent"):
        if c in rules_export.columns:
            rules_export[c] = rules_export[c].apply(
                lambda x: ",".join(sorted(str(i) for i in x))
                if isinstance(x, (list, tuple, frozenset, set, np.ndarray))
                else str(x))
    rules_export.to_csv(rules_path, index=False)
    log.info("EXPORT CSV       | %s (%s rows)", rules_path, f"{len(rules_export):,}")

    top_path = os.path.join(ASSOC_REPORTS_DIR, "top_rules_by_lift.csv")
    top_export = top_rules.copy()
    for c in ("antecedent", "consequent"):
        if c in top_export.columns:
            top_export[c] = top_export[c].apply(
                lambda x: ",".join(sorted(str(i) for i in x))
                if isinstance(x, (list, tuple, frozenset, set, np.ndarray))
                else str(x))
    top_export.to_csv(top_path, index=False)
    log.info("EXPORT CSV       | %s (%s rows)", top_path, f"{len(top_export):,}")


def export_parquets(fi_df, rules_filtered):
    fi_save = _serialize_listcol(fi_df, "itemset")
    fi_path = os.path.join(ASSOC_OUTPUT_PATH, "frequent_itemsets.parquet")
    fi_save.to_parquet(fi_path, index=False)
    log.info("EXPORT PARQUET   | %s", fi_path)

    rules_save = _serialize_listcol(rules_filtered, "antecedent")
    rules_save = _serialize_listcol(rules_save, "consequent")
    rules_path = os.path.join(ASSOC_OUTPUT_PATH, "association_rules.parquet")
    rules_save.to_parquet(rules_path, index=False)
    log.info("EXPORT PARQUET   | %s", rules_path)


# ---------------------------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------------------------

def plot_top_rules(top_rules, config):
    if len(top_rules) == 0:
        log.warning("PLOT             | top_rules empty -- skip top_rules_chart")
        return
    n_show = min(20, len(top_rules))
    plot_df = top_rules.head(n_show).copy()

    fig, ax = plt.subplots(figsize=(13, 8))
    colors = cm.RdYlGn(np.linspace(0.3, 0.9, n_show))
    ax.barh(
        range(n_show), plot_df["lift"].values[::-1],
        color=colors, edgecolor="white", height=0.75,
    )
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(plot_df["rule_str"].values[::-1], fontsize=7)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.2,
               label="lift=1 (no association)")
    ax.set_xlabel("Lift")
    ax.set_title(
        f"Top {n_show} Association Rules by Lift\n"
        f"(min_support={config['min_support']}, "
        f"min_confidence={config['min_confidence']}, "
        f"min_lift={config['min_lift']})",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ASSOC_REPORTS_DIR, "top_rules_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT             | %s", out)


def plot_support_distribution(fi_df):
    if len(fi_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(fi_df["support"], bins=40, color="#2166ac", alpha=0.85,
            edgecolor="white")
    median_v = fi_df["support"].median()
    ax.axvline(median_v, color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {median_v:.4f}")
    ax.set_xlabel("Support")
    ax.set_ylabel("Number of Itemsets")
    ax.set_title(f"Support Distribution of Frequent Itemsets "
                 f"(N={len(fi_df):,})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ASSOC_REPORTS_DIR, "support_dist_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT             | %s", out)


def plot_itemset_size_and_scatter(fi_df, top_rules, config):
    if len(fi_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: itemset size dist
    size_counts = fi_df["itemset_size"].value_counts().sort_index()
    axes[0].bar(size_counts.index, size_counts.values,
                color="#2166ac", alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Itemset Size (k)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Frequent Itemset Size Distribution")
    axes[0].grid(axis="y", alpha=0.3)
    for x, y in zip(size_counts.index, size_counts.values):
        axes[0].text(x, y + max(size_counts.values) * 0.01,
                     str(int(y)), ha="center", fontsize=9)

    # Right: confidence vs lift scatter
    if len(top_rules) > 0 and "support" in top_rules.columns:
        sc = axes[1].scatter(
            top_rules["confidence"], top_rules["lift"],
            c=top_rules["support"], cmap="RdYlGn", s=60, alpha=0.75,
            edgecolors="black", linewidths=0.4,
        )
        axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=1,
                        label="lift=1 baseline")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Lift")
        axes[1].set_title(f"Confidence vs Lift (top {len(top_rules)} rules)\n"
                          f"color = support")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[1], label="Support")
    else:
        axes[1].text(0.5, 0.5, "No rules with lift>=threshold",
                     ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    out = os.path.join(ASSOC_REPORTS_DIR, "itemset_size_dist.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT             | %s", out)


# ---------------------------------------------------------------------------
# SUMMARY TEXT
# ---------------------------------------------------------------------------

def write_summary_text(fi_df, rules_filtered, top_rules, config,
                        n_transactions, algo_name, fit_time):
    n_fi = len(fi_df)
    n_rules = len(rules_filtered)
    n_top = len(top_rules)
    min_sup = config["min_support"]
    min_conf = config["min_confidence"]
    min_lift = config["min_lift"]
    max_size = config["max_itemset_size"]
    avg_items = (fi_df["itemset_size"].mean()
                 if n_fi > 0 else 0.0)
    max_lift = (top_rules["lift"].max() if n_top > 0 else 0.0)
    avg_lift = (top_rules["lift"].mean() if n_top > 0 else 0.0)

    top5 = top_rules.head(5)
    top5_lines = []
    for _, r in top5.iterrows():
        top5_lines.append(
            f"    [#{r['rank']:2d}] {r['rule_str']}\n"
            f"           support={r['support']:.4f}  "
            f"confidence={r['confidence']:.4f}  lift={r['lift']:.4f}"
        )
    top5_str = "\n".join(top5_lines) if top5_lines else "    (no rules)"

    cols_used = list(config["assoc_cols"])
    cols_str = ", ".join(cols_used)

    text = f"""ASSOCIATION RULES SUMMARY -- MODULE 4 (SON / FPGrowth)
{"=" * 70}

[PASTE 1 -- Muc 6.1 Ma hoa transaction]

De ap dung khai pha luat ket hop, chung toi ma hoa moi cap (stock, ngay
giao dich) thanh mot transaction gom cac "items" la trang thai categorical
cua {len(cols_used)} chi bao ky thuat: {cols_str}. Voi tong so {n_transactions:,}
transactions ({n_transactions // 101 if n_transactions else 0} ngay x ~101 ma),
moi transaction chua trung binh {avg_items:.2f} items, tu bo tu vung gom cac
trang thai ky thuat nhu rsi_Oversold, bb_LOWER_ZONE, macd_BULLISH_CROSS,
adx_STRONG_TREND, trend_UP, vol_HIGH...

[PASTE 2 -- Muc 6.2 Thuat toan SON / FPGrowth]

Chung toi ap dung thuat toan SON (Savasere et al., 1995) thong qua trien
khai {algo_name}. Ve ban chat, day la SON 2-pass: Pass 1 mine local
frequent itemsets tren tung Spark partition voi nguong support dieu chinh
theo kich thuoc partition; Pass 2 aggregate va dem exact support tren
toan dataset de loai bo false positives.

Tham so su dung:
  - min_support     = {min_sup}  ({min_sup * 100:.1f}% transactions = ~{int(n_transactions * min_sup):,} ngay)
  - min_confidence  = {min_conf}  (P(Y|X) >= {min_conf * 100:.0f}%)
  - min_lift        = {min_lift}  (P(Y|X) / P(Y) >= {min_lift})
  - max_itemset_len = {max_size}  (gioi han combinatorial explosion)
  - thoi gian fit   = {fit_time:.1f}s

Ket qua: tim duoc {n_fi:,} frequent itemsets (max size <= {max_size}),
sinh ra {n_rules:,} association rules co lift >= {min_lift}. Lift trung binh
{avg_lift:.3f}, lift cao nhat {max_lift:.3f}.

[PASTE 3 -- Muc 6.3 Top rules va dien giai]

Top 5 luat ket hop theo lift:

{top5_str}

Dien giai: Lift > 1 cho thay antecedent va consequent co lien he duong
thuc su (khong phai ngau nhien). Vi du, rule dau bang (lift={max_lift:.3f})
cho thay khi nhom chi bao ben trai xuat hien dong thoi, nhom ben phai
co kha nang xuat hien cao hon {max_lift:.2f} lan so voi tan suat ngau
nhien. Day la insight de xay dung trading signal tong hop:
  - Confidence cao + lift cao  : rule dang tin cay, dung de phat tin hieu
  - Lift cao nhung support thap: pattern hiem nhung rat dac trung
  - Lift gan 1                 : antecedent va consequent gan nhu doc lap

[KET LUAN MODULE 4]

Su dung {algo_name}, chung toi chuyen tu cau hoi dinh tinh "Khi RSI
Oversold + BB Lower xay ra, dieu gi co xu huong xay ra cung?" thanh
{n_rules:,} luat dinh luong co the kiem chung. Trong do, {n_top} rules
dat nguong lift >= {min_lift} la nhung "trading signals" dung de chuyen
sang Dashboard cho viec test thuc nghiem chien luoc. Day la 1 trong 4
modul cua Phase 2 (cung voi PCA Clustering, LSH, Graph PageRank) ket
hop tao nen mot he thong phan tich co phieu da chieu hoan chinh.
"""

    out = os.path.join(ASSOC_REPORTS_DIR, "assoc_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("EXPORT TEXT      | %s", out)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Module 4: Association Rules (SON)")
    parser.add_argument("--no-spark", action="store_true",
                        help="Use mlxtend Apriori thay Spark FPGrowth (baseline)")
    parser.add_argument("--min-support", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-lift", type=float, default=None)
    parser.add_argument("--sample", type=float, default=None,
                        help="Sample fraction (0..1) cho quick test")
    parser.add_argument("--input-daily", default=None,
                        help=f"Path toi daily parquet (default: {LSH_PATH}/daily)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MODULE 4: ASSOCIATION RULES (SON / FPGrowth)")
    log.info("=" * 70)

    config = ASSOC_CONFIG.copy()
    if args.min_support is not None:
        config["min_support"] = args.min_support
    if args.min_confidence is not None:
        config["min_confidence"] = args.min_confidence
    if args.min_lift is not None:
        config["min_lift"] = args.min_lift
    if args.no_spark:
        config["use_spark"] = False

    assoc_cols = list(FEATURE_GROUPS.get("association_items", []))
    if not assoc_cols:
        log.error("FEATURE_GROUPS['association_items'] is empty in settings.py")
        sys.exit(1)
    config["assoc_cols"] = assoc_cols

    daily_path = args.input_daily or os.path.join(LSH_PATH, "daily")
    if not os.path.isdir(daily_path):
        log.error("Missing daily parquet at %s", daily_path)
        log.error("Run `python src/run_lsh.py` first to build daily aggregates")
        sys.exit(1)

    log.info("Daily input  : %s", daily_path)
    log.info("Reports dir  : %s", ASSOC_REPORTS_DIR)
    log.info("Output dir   : %s", ASSOC_OUTPUT_PATH)
    log.info("Algorithm    : %s",
             "Spark FPGrowth (SON-style)" if config["use_spark"]
             else "mlxtend Apriori (single-machine)")
    log.info("min_support=%.4f | min_confidence=%.4f | min_lift=%.2f | max_size=%d",
             config["min_support"], config["min_confidence"],
             config["min_lift"], config["max_itemset_size"])
    log.info("Columns (%d): %s", len(assoc_cols), assoc_cols)
    log.info("-" * 70)

    t_total = time.time()
    spark = None
    try:
        # STEP 1+2: Load + encode
        trans_df = load_and_encode_transactions(daily_path, assoc_cols, config)

        if args.sample and 0 < args.sample < 1:
            trans_df = trans_df.sample(frac=args.sample, random_state=42).reset_index(drop=True)
            log.info("STAGE 1 SAMPLE   | %.0f%% sample -> %s transactions",
                     args.sample * 100, f"{len(trans_df):,}")

        n_transactions = len(trans_df)
        if n_transactions == 0:
            log.error("No transactions after encoding. Check column data quality.")
            sys.exit(1)

        # STEP 3: Save transactions
        tx_path = os.path.join(ASSOC_OUTPUT_PATH, "transactions.parquet")
        trans_df.to_parquet(tx_path, index=False)
        log.info("STAGE 1 SAVE     | %s (%s rows)", tx_path, f"{n_transactions:,}")

        # STEP 4: Mine
        algo_name = None
        fit_time = 0.0
        if config["use_spark"]:
            spark = _create_spark(config)
            t0 = time.time()
            fi_df, rules_df = run_fpgrowth_spark(
                trans_df, spark,
                min_support=config["min_support"],
                min_confidence=config["min_confidence"],
                max_itemset_size=config["max_itemset_size"],
            )
            fit_time = time.time() - t0
            algo_name = "Spark MLlib FPGrowth (SON-style distributed)"
        else:
            t0 = time.time()
            fi_df, rules_df = run_apriori_pandas(
                trans_df,
                min_support=config["min_support"],
                min_confidence=config["min_confidence"],
                max_itemset_size=config["max_itemset_size"],
            )
            fit_time = time.time() - t0
            algo_name = "mlxtend Apriori (single-machine baseline)"

        if fi_df is None or len(fi_df) == 0:
            log.error("No frequent itemsets found. Try lower --min-support, vd: %.4f",
                      config["min_support"] / 2)
            sys.exit(1)

        # STEP 5: Filter + format
        rules_filtered, top_rules = filter_and_format_rules(
            rules_df, fi_df,
            min_lift=config["min_lift"],
            top_n=config["top_n_rules"],
        )

        # STEP 6: Export
        export_csvs(fi_df, rules_filtered, top_rules)
        export_parquets(fi_df, rules_filtered)
        plot_top_rules(top_rules, config)
        plot_support_distribution(fi_df)
        plot_itemset_size_and_scatter(fi_df, top_rules, config)

        # STEP 7: Summary
        write_summary_text(fi_df, rules_filtered, top_rules, config,
                            n_transactions, algo_name, fit_time)

        elapsed = time.time() - t_total
        log.info("=" * 70)
        log.info("MODULE 4 COMPLETED in %.1fs (%.1f min)", elapsed, elapsed / 60)
        log.info("=" * 70)
        log.info("Algorithm     : %s", algo_name)
        log.info("Transactions  : %s", f"{n_transactions:,}")
        log.info("Itemsets      : %s | Rules (lift>=%.2f): %s | Top: %s",
                 f"{len(fi_df):,}", config["min_lift"],
                 f"{len(rules_filtered):,}", f"{len(top_rules):,}")
        if len(top_rules) > 0:
            log.info("Top rule lift : %.4f", top_rules["lift"].iloc[0])
        log.info("=" * 70)

    except Exception:
        log.exception("MODULE 4 FAILED")
        sys.exit(1)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
