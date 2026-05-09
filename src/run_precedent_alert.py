"""
PHƯƠNG ÁN A — Runner: Cảnh báo theo tiền lệ (Similarity Search).

Pipeline:
    BUILD (1 lần, khi dữ liệu thay đổi):
        - Tạo feature vectors cho Ấn Độ + VN (STEP 2)
        - Dựng SimHash index cho Ấn Độ (STEP 3)
    QUERY (mỗi lần cảnh báo):
        - Lấy pattern 20 ngày hiện tại của mã VN (STEP 5 input)
        - Tìm tiền lệ Ấn Độ qua LSH + VN qua cosine exact (STEP 5–6)
        - Thống kê có base rate + khoảng tin cậy
        - Hợp nhất → risk score + nhãn cảnh báo (STEP 7)

Usage:
    conda activate stock

    # Build toàn bộ (cần có stock-market-data-final + stock-market-data-vn-final)
    PYTHONUNBUFFERED=1 python src/run_precedent_alert.py --build

    # Cảnh báo cho 1 mã VN tại 1 ngày (sau khi đã build)
    python src/run_precedent_alert.py --stock VCB --date 2026-06-05

    # Build xong query luôn
    python src/run_precedent_alert.py --build --stock FPT --date 2026-06-05 --top-k 50
"""

import os
import sys
import json
import time
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

# Workers can import 'similarity' module khi deserialize UDF
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = SRC_DIR + (os.pathsep + _existing if _existing else "")

import findspark
findspark.init()

from config.settings import (
    FINAL_PATH, VN_FINAL_PATH, PRECEDENT_PATH, PRECEDENT_REPORTS_DIR,
    PRECEDENT_CONFIG, SPARK_CONFIG, LOGS_DIR,
)
from similarity import (
    build_feature_vectors,
    build_simhash_index,
    load_hyperplanes,
    SimHasher,
    lookup_pattern_vector,
    query_precedents_lsh,
    query_precedents_exact,
    summarize_precedents,
    fuse_signals,
    run_sanity_check,
    run_backtest,
)
from similarity.precedent_query import compute_base_rate

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PRECEDENT_REPORTS_DIR, exist_ok=True)

log = logging.getLogger("precedent")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "precedent_alert.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)

# Duong dan artifacts
INDIA_DIR = os.path.join(PRECEDENT_PATH, "india")
VN_DIR = os.path.join(PRECEDENT_PATH, "vn")
INDIA_VECTORS = os.path.join(INDIA_DIR, "vectors")
VN_VECTORS = os.path.join(VN_DIR, "vectors")
INDIA_INDEX = os.path.join(INDIA_DIR, "simhash-index")


def _create_spark():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("PrecedentAlert")
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
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def run_build(spark):
    log.info("-" * 65)
    log.info("BUILD | STEP 2: feature vectors (India + VN)")
    if not os.path.isdir(FINAL_PATH):
        log.error("Thieu India final parquet: %s", FINAL_PATH)
        sys.exit(1)
    if not os.path.isdir(VN_FINAL_PATH):
        log.error("Thieu VN final parquet: %s "
                  "(chay: python src/main_pipeline.py --market vn)", VN_FINAL_PATH)
        sys.exit(1)

    build_feature_vectors(spark, FINAL_PATH, INDIA_DIR, "india", PRECEDENT_CONFIG)
    build_feature_vectors(spark, VN_FINAL_PATH, VN_DIR, "vn", PRECEDENT_CONFIG)

    log.info("BUILD | STEP 3: SimHash index (India)")
    build_simhash_index(spark, INDIA_VECTORS, INDIA_DIR, PRECEDENT_CONFIG)
    log.info("BUILD | Done.")


# ---------------------------------------------------------------------------
# QUERY
# ---------------------------------------------------------------------------

def run_query(spark, stock: str, date: str, top_k: int) -> dict:
    log.info("-" * 65)
    log.info("QUERY | stock=%s date=%s top_k=%d", stock, date, top_k)

    # 1) Pattern hien tai cua ma VN
    q = lookup_pattern_vector(spark, VN_VECTORS, stock, date)
    if q is None:
        log.error("Khong tim thay pattern cho %s tai/<= %s trong VN vectors.", stock, date)
        sys.exit(1)
    qvec = q["vector"]
    actual_date = q["trade_date"]
    log.info("QUERY | dung pattern VN %s @ %s (close=%.2f)",
             stock, actual_date, q["close"])

    # 2) Tien le An Do qua LSH
    planes = load_hyperplanes(INDIA_DIR)
    hasher = SimHasher(planes, PRECEDENT_CONFIG["simhash_band_bits"],
                       PRECEDENT_CONFIG["simhash_n_bands"])
    buckets = hasher.buckets(qvec)
    india_prec = query_precedents_lsh(spark, INDIA_INDEX, qvec, buckets, top_k)

    # 3) Tien le VN qua cosine exact (chi dung qua khu -> chong look-ahead)
    vn_prec = query_precedents_exact(
        spark, VN_VECTORS, qvec, top_k,
        query_stock=stock, before_date=actual_date,
    )

    # 4) Base rate + thong ke
    india_base = compute_base_rate(spark, INDIA_VECTORS)
    vn_base = compute_base_rate(spark, VN_VECTORS)
    ci_z = PRECEDENT_CONFIG["ci_z"]
    india_sum = summarize_precedents(india_prec, india_base, ci_z)
    vn_sum = summarize_precedents(vn_prec, vn_base, ci_z)

    # 5) Hop nhat
    verdict = fuse_signals(india_sum, vn_sum, PRECEDENT_CONFIG)

    result = {
        "query": {"stock": stock, "requested_date": date,
                  "pattern_date": actual_date, "top_k": top_k},
        "india_signal": india_sum,
        "vn_signal": vn_sum,
        "verdict": verdict,
        "india_precedents": india_prec[:10],
        "vn_precedents": vn_prec[:10],
    }
    _print_verdict(result)
    _save_report(result, stock, actual_date)
    return result


def _print_verdict(r: dict):
    v = r["verdict"]
    print("\n" + "=" * 65)
    print(f"  CẢNH BÁO: {v['label']}   |   Risk score: {v['risk_score']}/100")
    print(f"  Độ tin cậy: {v['confidence_label']} ({v['confidence']})  "
          f"|  {v['agreement']}")
    print("=" * 65)
    for line in v["reason"]:
        print("  - " + line)
    print("=" * 65 + "\n")


def _save_report(r: dict, stock: str, date: str):
    fname = f"alert_{stock}_{date}.json"
    path = os.path.join(PRECEDENT_REPORTS_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False, default=str)
    log.info("QUERY | Report -> %s", path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phương án A: Cảnh báo theo tiền lệ")
    parser.add_argument("--build", action="store_true",
                        help="Build feature vectors + SimHash index")
    parser.add_argument("--sanity", action="store_true",
                        help="STEP 4: sanity check phân phối Ấn Độ vs VN")
    parser.add_argument("--backtest", action="store_true",
                        help="STEP 8: backtest walk-forward (Spark LSH-join)")
    parser.add_argument("--stock", default=None, help="Mã VN cần cảnh báo (vd VCB)")
    parser.add_argument("--date", default=None, help="Ngày phân tích YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=PRECEDENT_CONFIG["top_k"],
                        help=f"Số tiền lệ giống nhất (default {PRECEDENT_CONFIG['top_k']})")
    args = parser.parse_args()

    if not any([args.build, args.sanity, args.backtest, args.stock]):
        parser.error(
            "Cần một trong: --build | --sanity | --backtest | --stock <MÃ> --date <NGÀY>"
        )

    log.info("=" * 65)
    log.info("PHUONG AN A: CANH BAO THEO TIEN LE (cosine + SimHash LSH)")
    log.info("=" * 65)

    spark = _create_spark()
    log.info("Spark: %s | cores: %d", spark.version, spark.sparkContext.defaultParallelism)
    t0 = time.time()

    try:
        if args.build:
            run_build(spark)
        if args.sanity:
            run_sanity_check(spark, FINAL_PATH, VN_FINAL_PATH,
                             INDIA_VECTORS, VN_VECTORS,
                             PRECEDENT_CONFIG, PRECEDENT_REPORTS_DIR)
        if args.backtest:
            run_backtest(spark, INDIA_DIR, VN_DIR, PRECEDENT_CONFIG,
                         PRECEDENT_REPORTS_DIR)
        if args.stock:
            if not args.date:
                parser.error("--stock yeu cau di kem --date YYYY-MM-DD")
            run_query(spark, args.stock.upper(), args.date, args.top_k)

        log.info("DONE in %.1fs", time.time() - t0)
    except Exception:
        log.exception("PRECEDENT ALERT FAILED")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
