"""
STOCK BIG DATA - MAIN ETL PIPELINE

Entry point duy nhat: CSV tho -> Parquet cuoi cung (khong intermediate).

Usage:
    conda activate stock
    python src/main_pipeline.py

Input  : stock/stock-market-data/*.csv  (~60GB, 101 files, 60 columns)
Output : stock/stock-market-data-final/ (Parquet, partitioned year/month/stock_symbol)
"""

import os
import sys
import time
import logging
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofmonth, to_timestamp,
    input_file_name, regexp_extract, count,
)

from config.settings import (
    RAW_CSV_PATH, FINAL_PATH, SPARK_CONFIG, LOGS_DIR,
    get_csv_schema, FEATURE_GROUPS,
)
from features import (
    add_trend_features,
    add_momentum_features,
    add_volatility_features,
    add_phase2_features,
)


os.makedirs(LOGS_DIR, exist_ok=True)

log = logging.getLogger("stock_etl")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "main_pipeline.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


def create_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(SPARK_CONFIG["app_name"] + "_ETL")
        .master(SPARK_CONFIG["master"])
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
        .config("spark.driver.maxResultSize", SPARK_CONFIG["max_result_size"])
        .config("spark.sql.adaptive.enabled", SPARK_CONFIG["adaptive_enabled"])
        .config("spark.sql.adaptive.coalescePartitions.enabled",
                SPARK_CONFIG["coalesce_partitions_enabled"])
        .config("spark.memory.fraction", SPARK_CONFIG["spark.memory.fraction"])
        .config("spark.memory.storageFraction", SPARK_CONFIG["spark.memory.storageFraction"])
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG["spark.sql.shuffle.partitions"])
        # Window executor spill sang disk thay vi OOM khi partition lon
        .config("spark.sql.windowExec.buffer.spill.threshold",
                SPARK_CONFIG["window_spill_threshold"])
        .config("spark.sql.windowExec.buffer.in.memory.threshold",
                SPARK_CONFIG["window_spill_threshold"])
        .getOrCreate()
    )
    _suppress_spark_warnings(spark)
    return spark


def _suppress_spark_warnings(spark: SparkSession) -> None:
    """Tat cac WARN khong can thiet cua Spark 4.0."""
    try:
        log4j = spark.sparkContext._jvm.org.apache.log4j
        # Spark 4.0 kiem tra streaming metadata khi doc CSV glob -> WARN harmless
        log4j.Logger.getLogger(
            "org.apache.spark.sql.execution.streaming.FileStreamSink"
        ).setLevel(log4j.Level.ERROR)
    except Exception:
        pass


# =========================================================================
# PIPELINE STEPS
# =========================================================================

def step_extract(spark: SparkSession):
    """EXTRACT: Doc 101 file CSV tho voi schema co dinh."""
    csv_pattern = os.path.join(RAW_CSV_PATH, "*.csv")
    csv_files = glob.glob(csv_pattern)
    log.info("STEP 1 EXTRACT  | %d CSV files from %s", len(csv_files), RAW_CSV_PATH)

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .schema(get_csv_schema())
        .csv(csv_pattern)
    )
    return df


def step_enrich(df):
    """ENRICH: Them stock_symbol tu ten file, parse timestamp, tao partition keys."""
    log.info("STEP 2 ENRICH   | Adding metadata columns")

    df = df.withColumn("_fp", input_file_name())
    df = df.withColumn(
        "stock_symbol",
        regexp_extract(col("_fp"), r"([^/]+)_minute_data_with_indicators\.csv$", 1),
    )
    df = df.drop("_fp")

    df = df.withColumn("timestamp", to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ssXXX"))
    df = df.withColumn("year", year(col("timestamp")))
    df = df.withColumn("month", month(col("timestamp")))
    df = df.withColumn("day", dayofmonth(col("timestamp")))
    return df


def step_clean(df):
    """CLEAN: Loai bo rows thieu du lieu critical (close, timestamp, stock_symbol)."""
    log.info("STEP 3 CLEAN    | Dropping rows with null critical fields")

    df_clean = df.filter(
        col("close").isNotNull()
        & col("timestamp").isNotNull()
        & col("stock_symbol").isNotNull()
        & (col("stock_symbol") != "")
        & (col("close") > 0)
    )
    return df_clean


def step_transform(df):
    """TRANSFORM: Feature engineering 4 nhom (trend, momentum, volatility, phase2)."""
    log.info("STEP 4 TRANSFORM | 4A Trend features")
    df = add_trend_features(df)

    log.info("STEP 4 TRANSFORM | 4B Momentum features")
    df = add_momentum_features(df)

    log.info("STEP 4 TRANSFORM | 4C Volatility features")
    df = add_volatility_features(df)

    # Repartition theo stock_symbol TRUOC khi chay Window operations trong phase2.
    # Phase2 dung Window.partitionBy("stock_symbol") cho lag() => Spark can
    # shuffle du lieu theo stock. Repartition truoc giup:
    # - Du lieu 1 stock nam gon tren 1 partition => Window khong can shuffle lai
    # - Moi partition nho hon (~628K rows thay vi hang trieu) => it memory hon
    log.info("STEP 4 TRANSFORM | Repartitioning by stock_symbol for Window operations")
    df = df.repartition("stock_symbol")

    log.info("STEP 4 TRANSFORM | 4D Phase 2 prep features")
    df = add_phase2_features(df)

    return df


def step_load(df):
    """LOAD: Ghi Parquet partitioned theo year/month/stock_symbol."""
    log.info("STEP 5 LOAD     | Writing to %s", FINAL_PATH)
    log.info("STEP 5 LOAD     | Partition by: year / month / stock_symbol")

    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month", "stock_symbol")
        .option("compression", SPARK_CONFIG["compression_codec"])
        .parquet(FINAL_PATH)
    )
    log.info("STEP 5 LOAD     | Write complete")


def step_verify(spark: SparkSession):
    """VERIFY: Doc lai Parquet, kiem tra row count, schema, partition pruning."""
    log.info("STEP 6 VERIFY   | Reading back Parquet from %s", FINAL_PATH)

    df = spark.read.parquet(FINAL_PATH)
    total = df.count()
    log.info("STEP 6 VERIFY   | Total rows: %s | Total cols: %d", f"{total:,}", len(df.columns))

    nulls = df.select(
        count(col("log_return")).alias("log_return"),
        count(col("next_3d_label")).alias("next_3d_label"),
        count(col("rsi_status")).alias("rsi_status"),
        count(col("bb_position_label")).alias("bb_position_label"),
        count(col("close_zscore")).alias("close_zscore"),
    ).collect()[0]
    log.info(
        "STEP 6 VERIFY   | Non-null counts: log_return=%s next_3d_label=%s "
        "rsi_status=%s bb_position_label=%s close_zscore=%s",
        f"{nulls['log_return']:,}",
        f"{nulls['next_3d_label']:,}",
        f"{nulls['rsi_status']:,}",
        f"{nulls['bb_position_label']:,}",
        f"{nulls['close_zscore']:,}",
    )

    df.createOrReplaceTempView("stocks_final")

    t0 = time.time()
    r1 = spark.sql(
        "SELECT COUNT(*) c FROM stocks_final "
        "WHERE year=2022 AND month=1 AND stock_symbol='HDFC'"
    ).collect()
    log.info(
        "STEP 6 VERIFY   | Partition query (1 stock, 1 month): %s rows in %.3fs",
        f"{r1[0]['c']:,}", time.time() - t0,
    )

    t0 = time.time()
    r2 = spark.sql(
        "SELECT stock_symbol, COUNT(*) c FROM stocks_final "
        "WHERE year=2022 GROUP BY stock_symbol ORDER BY c DESC"
    ).collect()
    log.info(
        "STEP 6 VERIFY   | Cross-stock query (all stocks, 1 year): %d stocks in %.3fs",
        len(r2), time.time() - t0,
    )

    log.info("STEP 6 VERIFY   | Schema: %s", df.columns)
    log.info(
        "STEP 6 VERIFY   | Feature groups available: pca_input=%d cols, "
        "lsh_fingerprint=%d cols, association_items=%d cols",
        len(FEATURE_GROUPS["pca_input"]),
        len(FEATURE_GROUPS["lsh_fingerprint"]),
        len(FEATURE_GROUPS["association_items"]),
    )
    return df


# =========================================================================
# MAIN
# =========================================================================

def main():
    total_start = time.time()
    log.info("=" * 65)
    log.info("STOCK BIG DATA - MAIN ETL PIPELINE")
    log.info("=" * 65)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")
    log.info("Spark version: %s | cores: %d", spark.version, spark.sparkContext.defaultParallelism)

    try:
        df = step_extract(spark)
        df = step_enrich(df)
        df = step_clean(df)
        df = step_transform(df)
        step_load(df)
        step_verify(spark)

        elapsed = time.time() - total_start
        log.info("=" * 65)
        log.info("PIPELINE COMPLETED in %.1f min (%.0f sec)", elapsed / 60, elapsed)
        log.info("=" * 65)

    except Exception:
        log.exception("PIPELINE FAILED")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
