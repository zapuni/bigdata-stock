"""
LSH Stock Similarity -- CS246: Shingling -> MinHashing -> LSH Banding

Bai toan: Tim cac co phieu co pattern giao dich tuong tu trong N ngay gan nhat.

Pipeline:
  STAGE 1  daily_agg        : 63M rows (1-min) -> ~250K rows (1-day snapshot)
  STAGE 2  shingling_minhash: ~250K rows -> signature matrix (250K x 100)
  STAGE 3  lsh_banding      : signature matrix -> candidate pairs
  QUERY    query_similar     : Tim top-K stocks tuong tu voi 1 stock tai 1 ngay

Input : stock-market-data-final/ (ETL pipeline output)
Output: lsh-similarity/ (daily/, signatures/, candidate-pairs/)
"""

import os
import time
import hashlib
import logging

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, last, concat_ws, to_date, size, when,
    collect_list, count, posexplode,
)
from pyspark.sql.types import ArrayType, LongType, IntegerType
from pyspark.sql.window import Window
from pyspark.sql import functions as F

log = logging.getLogger("stock_lsh")

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------
WINDOW_DAYS = 20
K_SHINGLE = 2
N_HASH = 100
N_BANDS = 20
ROWS_PER_BAND = N_HASH // N_BANDS  # 5

# Discrete signal columns (lsh_fingerprint group tu settings.py)
SIGNAL_COLS = [
    "rsi_status",
    "macd_cross_signal",
    "bb_position_label",
    "adx_strength_label",
    "trend_ema_cross",
    "is_high_volatility",
]

# MinHash parameters (fixed seed cho reproducible)
_RNG = np.random.RandomState(42)
_LARGE_PRIME = (1 << 31) - 1
_A = _RNG.randint(1, _LARGE_PRIME, size=N_HASH).tolist()
_B = _RNG.randint(0, _LARGE_PRIME, size=N_HASH).tolist()


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _hash_shingle(shingle: str) -> int:
    return int(hashlib.md5(shingle.encode()).hexdigest()[:8], 16)


def _create_shingle_set(states: list, k: int) -> set:
    shingles = set()
    for i in range(len(states) - k + 1):
        shingle_str = " -> ".join(states[i : i + k])
        shingles.add(_hash_shingle(shingle_str))
    return shingles


def _minhash_signature(shingle_set: set) -> list:
    """MinHash_i = min over all x in set of h_i(x), h_i(x) = (A[i]*x + B[i]) mod P."""
    if not shingle_set:
        return [0] * N_HASH
    sig = []
    for i in range(N_HASH):
        min_val = _LARGE_PRIME
        for x in shingle_set:
            hv = (_A[i] * x + _B[i]) % _LARGE_PRIME
            if hv < min_val:
                min_val = hv
        sig.append(int(min_val))
    return sig


def _band_hash(band_values: list) -> int:
    s = "|".join(str(v) for v in band_values)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# UDFs (registered lazily via _register_udfs)
# ---------------------------------------------------------------------------
_UDFS_REGISTERED = False


def _register_udfs(spark: SparkSession):
    global _UDFS_REGISTERED
    if _UDFS_REGISTERED:
        return
    spark.udf.register("compute_signature", _udf_compute_signature)
    spark.udf.register("compute_band_hashes", _udf_compute_band_hashes)
    _UDFS_REGISTERED = True


@F.udf(returnType=ArrayType(LongType()))
def _udf_compute_signature(state_list):
    if not state_list or len(state_list) < K_SHINGLE:
        return [0] * N_HASH
    shingle_set = _create_shingle_set(state_list, K_SHINGLE)
    return _minhash_signature(shingle_set)


@F.udf(returnType=ArrayType(IntegerType()))
def _udf_compute_band_hashes(signature):
    if not signature:
        return [0] * N_BANDS
    band_hashes = []
    for b in range(N_BANDS):
        start = b * ROWS_PER_BAND
        band_chunk = signature[start : start + ROWS_PER_BAND]
        band_hashes.append(_band_hash(band_chunk))
    return band_hashes


# ---------------------------------------------------------------------------
# STAGE 1: DAILY AGGREGATION
# ---------------------------------------------------------------------------

def stage1_daily_agg(spark: SparkSession, final_path: str, output_dir: str) -> str:
    """63.9M rows (1-min) -> ~250K rows (1-day snapshot cuoi ngay).

    Args:
        spark: SparkSession
        final_path: duong dan toi ETL final parquet
        output_dir: thu muc goc cua LSH artifacts

    Returns:
        duong dan toi daily parquet
    """
    log.info("STAGE 1 DAILY AGG | Reading from %s", final_path)
    t0 = time.time()

    df = spark.read.parquet(final_path)

    # Trich trade_date tu timestamp (cot minute-level -> date-level)
    df = df.withColumn("trade_date", to_date(col("timestamp")))

    agg_exprs = [last(c, ignorenulls=True).alias(c) for c in SIGNAL_COLS]
    agg_exprs += [
        last("close", ignorenulls=True).alias("close"),
        last("log_return", ignorenulls=True).alias("log_return"),
        last("next_3d_label", ignorenulls=True).alias("next_3d_label"),
    ]

    daily = (
        df.groupBy("stock_symbol", "year", "month", "trade_date")
        .agg(*agg_exprs)
        .orderBy("stock_symbol", "trade_date")
    )

    out = os.path.join(output_dir, "daily")
    daily.write.mode("overwrite").partitionBy("year", "stock_symbol").parquet(out)

    total = daily.count()
    log.info(
        "STAGE 1 DAILY AGG | %s rows in %.1fs -> %s",
        f"{total:,}", time.time() - t0, out,
    )
    return out


# ---------------------------------------------------------------------------
# STAGE 2: SHINGLING + MINHASHING
# ---------------------------------------------------------------------------

def stage2_shingling_minhash(spark: SparkSession, daily_path: str, output_dir: str) -> str:
    """Daily data -> signature matrix (250K x 100 MinHash values).

    Moi (stock, trade_date) nhin lai 20 ngay, tao 2-shingle set, tinh MinHash.
    """
    log.info(
        "STAGE 2 MINHASH  | window=%d days, k=%d-shingle, %d hashes",
        WINDOW_DAYS, K_SHINGLE, N_HASH,
    )
    t0 = time.time()

    _register_udfs(spark)
    df = spark.read.parquet(daily_path)

    # Tao daily_state: ghep tat ca signal cols thanh 1 string
    df = df.withColumn("daily_state", concat_ws("|", *[col(c) for c in SIGNAL_COLS]))

    # Rolling window 20 ngay: collect_list daily_state
    w = (
        Window.partitionBy("stock_symbol")
        .orderBy("trade_date")
        .rowsBetween(-WINDOW_DAYS + 1, 0)
    )
    df = df.withColumn("state_window", collect_list("daily_state").over(w))

    # Chi giu rows du 20 ngay (bo warm-up)
    df = df.filter(size("state_window") == WINDOW_DAYS)

    # MinHash UDF
    log.info("STAGE 2 MINHASH  | Computing signatures...")
    df = df.withColumn("signature", _udf_compute_signature("state_window"))
    df = df.drop("state_window", "daily_state")

    out = os.path.join(output_dir, "signatures")
    df.write.mode("overwrite").partitionBy("year", "stock_symbol").parquet(out)

    log.info("STAGE 2 MINHASH  | Done in %.1fs -> %s", time.time() - t0, out)
    return out


# ---------------------------------------------------------------------------
# STAGE 3: LSH BANDING
# ---------------------------------------------------------------------------

def stage3_lsh_banding(spark: SparkSession, sig_path: str, output_dir: str) -> str:
    """Signature matrix -> candidate pairs via banding.

    b=20 bands x r=5 rows, threshold t ~ (1/b)^(1/r) ~ 0.55.
    Hai stocks co >= 1 band match -> candidate pair.
    """
    threshold = (1 / N_BANDS) ** (1 / ROWS_PER_BAND)
    log.info(
        "STAGE 3 BANDING  | b=%d bands, r=%d rows, threshold~%.2f",
        N_BANDS, ROWS_PER_BAND, threshold,
    )
    t0 = time.time()

    _register_udfs(spark)
    df = spark.read.parquet(sig_path)

    df = df.withColumn("band_hashes", _udf_compute_band_hashes("signature"))

    df_bands = df.select(
        "stock_symbol",
        "trade_date",
        "year",
        "close",
        "log_return",
        "next_3d_label",
        posexplode("band_hashes").alias("band_id", "bucket_hash"),
    )

    df_a = df_bands.alias("a")
    df_b = df_bands.alias("b")

    candidates = (
        df_a.join(
            df_b,
            (col("a.band_id") == col("b.band_id"))
            & (col("a.bucket_hash") == col("b.bucket_hash"))
            & (col("a.stock_symbol") < col("b.stock_symbol")),
        )
        .select(
            col("a.stock_symbol").alias("stock_a"),
            col("a.trade_date").alias("date_a"),
            col("b.stock_symbol").alias("stock_b"),
            col("b.trade_date").alias("date_b"),
            col("a.band_id"),
        )
        .groupBy("stock_a", "date_a", "stock_b", "date_b")
        .agg(count("band_id").alias("bands_matched"))
        .orderBy("bands_matched", ascending=False)
    )

    out = os.path.join(output_dir, "candidate-pairs")
    candidates.write.mode("overwrite").parquet(out)

    total = candidates.count()
    log.info(
        "STAGE 3 BANDING  | %s candidate pairs in %.1fs -> %s",
        f"{total:,}", time.time() - t0, out,
    )
    return out


# ---------------------------------------------------------------------------
# QUERY
# ---------------------------------------------------------------------------

def query_similar(
    spark: SparkSession,
    output_dir: str,
    query_stock: str,
    query_date: str,
    top_k: int = 5,
):
    """Tim top-K stocks co pattern 20-ngay tuong tu voi query_stock tai query_date.

    Logic:
    - Filter candidate pairs co (query_stock, query_date) o mot trong hai chieu.
    - Neu khong co pair nao cho ngay cu the, fallback sang ngay co nhieu pair nhat
      cua query_stock de demo.
    - Aggregate theo similar_stock: giu ngay co bands_matched cao nhat, tranh
      cung 1 stock xuat hien nhieu lan trong ket qua.

    Returns:
        Spark DataFrame: similar_stock, best_match_date, bands_matched,
        close, next_3d_label, va cac signal cols.
    """
    pairs_path = os.path.join(output_dir, "candidate-pairs")
    daily_path = os.path.join(output_dir, "daily")

    df = spark.read.parquet(pairs_path)

    def _filter_by_date(pairs_df, stock, date):
        return pairs_df.filter(
            ((col("stock_a") == stock) & (col("date_a") == date))
            | ((col("stock_b") == stock) & (col("date_b") == date))
        )

    filtered = _filter_by_date(df, query_stock, query_date)
    actual_date = query_date

    if filtered.rdd.isEmpty():
        # Fallback: tim ngay co nhieu pair nhat cua stock nay
        log.warning(
            "QUERY | No pairs for (%s, %s). Finding best available date...",
            query_stock, query_date,
        )
        all_stock_pairs = df.filter(
            (col("stock_a") == query_stock) | (col("stock_b") == query_stock)
        )
        if all_stock_pairs.rdd.isEmpty():
            log.warning("QUERY | Stock %s has no candidate pairs at all.", query_stock)
            return spark.createDataFrame([], schema=df.schema)

        best = (
            all_stock_pairs
            .withColumn(
                "qdate",
                when(col("stock_a") == query_stock, col("date_a")).otherwise(col("date_b")),
            )
            .groupBy("qdate")
            .agg(F.count("*").alias("n"))
            .orderBy("n", ascending=False)
            .limit(1)
            .collect()
        )
        actual_date = str(best[0]["qdate"])
        log.info(
            "QUERY | Fallback date: %s (%d pairs for %s)",
            actual_date, best[0]["n"], query_stock,
        )
        filtered = _filter_by_date(df, query_stock, actual_date)

    # Normalize: luon dat query stock vao mot chieu
    normalized = filtered.withColumn(
        "similar_stock",
        when(col("stock_a") == query_stock, col("stock_b")).otherwise(col("stock_a")),
    ).withColumn(
        "similar_date",
        when(col("stock_a") == query_stock, col("date_b")).otherwise(col("date_a")),
    )

    # Aggregate theo similar_stock: giu ban ghi co bands_matched cao nhat
    w = Window.partitionBy("similar_stock").orderBy(col("bands_matched").desc())
    top_results = (
        normalized
        .withColumn("_rank", F.row_number().over(w))
        .filter(col("_rank") == 1)
        .drop("_rank", "stock_a", "date_a", "stock_b", "date_b")
        .select("similar_stock", "similar_date", "bands_matched")
        .orderBy("bands_matched", ascending=False)
        .limit(top_k)
    )

    daily = spark.read.parquet(daily_path).select(
        col("stock_symbol"),
        col("trade_date"),
        col("close"),
        col("next_3d_label"),
        *[col(c) for c in SIGNAL_COLS],
    )

    enriched = top_results.join(
        daily,
        (top_results.similar_stock == daily.stock_symbol)
        & (top_results.similar_date == daily.trade_date),
        how="left",
    ).drop("stock_symbol", "trade_date")

    log.info(
        "QUERY | Top %d similar to %s on %s (b=%d, r=%d, threshold~%.2f)",
        top_k, query_stock, actual_date, N_BANDS, ROWS_PER_BAND,
        (1 / N_BANDS) ** (1 / ROWS_PER_BAND),
    )
    return enriched


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------

def benchmark(spark: SparkSession, sig_path: str, output_dir: str):
    """So sanh thoi gian LSH vs Brute-force (estimated)."""
    df = spark.read.parquet(sig_path)
    n = df.count()
    total_pairs = n * (n - 1) // 2

    t0 = time.time()
    candidates_path = os.path.join(output_dir, "candidate-pairs")
    lsh_pairs = spark.read.parquet(candidates_path).count()
    t_lsh = time.time() - t0

    log.info("BENCHMARK | Dataset: %s (stock, date) pairs", f"{n:,}")
    log.info("BENCHMARK | All pairs: %s", f"{total_pairs:,}")
    log.info(
        "BENCHMARK | LSH candidates: %s (%.4f%% of all pairs)",
        f"{lsh_pairs:,}", lsh_pairs / total_pairs * 100 if total_pairs > 0 else 0,
    )
    log.info(
        "BENCHMARK | Reduction: %sx fewer comparisons",
        f"{total_pairs // max(lsh_pairs, 1):,}",
    )
    log.info("BENCHMARK | LSH read time: %.2fs", t_lsh)
    log.info(
        "BENCHMARK | Brute-force estimated: %.1fs (1us/comparison)",
        total_pairs * 1e-6,
    )


# ---------------------------------------------------------------------------
# BUILD ALL (convenience)
# ---------------------------------------------------------------------------

def build_all(spark: SparkSession, final_path: str, output_dir: str):
    """Chay toan bo pipeline LSH: Stage 1 -> 2 -> 3 + benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    daily_path = stage1_daily_agg(spark, final_path, output_dir)
    sig_path = stage2_shingling_minhash(spark, daily_path, output_dir)
    stage3_lsh_banding(spark, sig_path, output_dir)
    benchmark(spark, sig_path, output_dir)

    log.info("BUILD ALL | Total: %.1f min", (time.time() - t0) / 60)
