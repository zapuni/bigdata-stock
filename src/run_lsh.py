"""
Runner for LSH Stock Similarity -- Phase 2

Usage:
    # Build toan bo (Stage 1-3 + benchmark)
    python src/run_lsh.py

    # Chi query (sau khi da build)
    python src/run_lsh.py --query-only --stock HDFC_BANK --date 2024-03-15

    # Build + query stock cu the
    python src/run_lsh.py --stock ICICIBANK --date 2023-06-01 --top 10
"""

import os
import sys
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

# Workers inherit env vars, not sys.path -> must export PYTHONPATH
# so cloudpickle can resolve 'algorithms' module in worker subprocesses
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = SRC_DIR + (os.pathsep + _existing if _existing else "")

import findspark
findspark.init()

from config.settings import FINAL_PATH, LSH_PATH, SPARK_CONFIG, LOGS_DIR
from algorithms.lsh import build_all, query_similar

os.makedirs(LOGS_DIR, exist_ok=True)

log = logging.getLogger("stock_lsh")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "lsh_pipeline.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


def _create_spark():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("LSH_StockSimilarity")
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
        # Ensure worker subprocesses can import src/ modules (UDF deserialization)
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    parser = argparse.ArgumentParser(description="LSH Stock Similarity Pipeline")
    parser.add_argument("--query-only", action="store_true",
                        help="Skip build, only run query")
    parser.add_argument("--stock", default="HDFCBANK",
                        help="Stock symbol to query (default: HDFCBANK)")
    parser.add_argument("--date", default="2022-10-20",
                        help="Trade date to query YYYY-MM-DD (default: 2022-10-20)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of similar stocks (default: 5)")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("LSH STOCK SIMILARITY PIPELINE")
    log.info("=" * 65)

    spark = _create_spark()
    log.info("Spark: %s | cores: %d", spark.version, spark.sparkContext.defaultParallelism)
    log.info("Input : %s", FINAL_PATH)
    log.info("Output: %s", LSH_PATH)

    try:
        if not args.query_only:
            build_all(spark, FINAL_PATH, LSH_PATH)

        result = query_similar(spark, LSH_PATH, args.stock, args.date, args.top)
        result.show(args.top, truncate=False)

    except Exception:
        log.exception("LSH PIPELINE FAILED")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
