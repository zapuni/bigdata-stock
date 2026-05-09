"""
STEP 5 + 6 — Truy vấn tiền lệ + thống kê (có base rate & khoảng tin cậy).

- query_precedents_lsh  : Ấn Độ → lọc candidate qua SimHash buckets rồi cosine.
- query_precedents_exact: VN    → cosine vét cạn (dữ liệu nhỏ, chính xác).
- summarize_precedents  : thống kê top-k (p_down, lợi suất TB, sâu nhất) so với
                          base rate của thị trường + khoảng tin cậy Wilson.

Tất cả dùng CÙNG cách đo cosine ⇒ so sánh công bằng giữa hai thị trường.
"""

import math
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType

log = logging.getLogger("precedent")


# ---------------------------------------------------------------------------
# COSINE CORE (query vector da L2-normalized => cosine = dot)
# ---------------------------------------------------------------------------

def _rank_by_cosine(spark: SparkSession, df, query_vector: list, top_k: int) -> list:
    """Tinh cosine giua query_vector va moi dong, lay top_k cao nhat.

    Yeu cau: df co cot `vector` (da L2-normalized) va `fwd_down` not null.
    """
    qv_bc = spark.sparkContext.broadcast([float(x) for x in query_vector])

    @F.udf(returnType=DoubleType())
    def _cosine_udf(vector):
        if not vector:
            return None
        qv = qv_bc.value
        if len(vector) != len(qv):
            return None
        return float(sum(a * b for a, b in zip(vector, qv)))

    ranked = (
        df.filter(col("fwd_down").isNotNull())
        .withColumn("cosine", _cosine_udf("vector"))
        .filter(col("cosine").isNotNull())
        .orderBy(col("cosine").desc())
        .limit(top_k)
        .select("stock_symbol", "trade_date", "cosine", "fwd_return", "fwd_down")
        .collect()
    )
    return [r.asDict() for r in ranked]


# ---------------------------------------------------------------------------
# LOOKUP: vector cua pattern hien tai (query)
# ---------------------------------------------------------------------------

def lookup_pattern_vector(spark: SparkSession, vectors_dir: str, stock: str, date: str):
    """Lay vector cua (stock, date). Neu date khong co -> ngay gan nhat <= date.

    Returns:
        dict {stock, trade_date, close, vector} hoac None neu khong co.
    """
    df = spark.read.parquet(vectors_dir).filter(col("stock_symbol") == stock)
    if df.rdd.isEmpty():
        return None

    exact = df.filter(col("trade_date") == lit(date))
    if not exact.rdd.isEmpty():
        row = exact.first()
    else:
        prior = df.filter(col("trade_date") <= lit(date)).orderBy(col("trade_date").desc())
        if prior.rdd.isEmpty():
            log.warning("LOOKUP | %s khong co pattern <= %s", stock, date)
            return None
        row = prior.first()
        log.warning("LOOKUP | %s khong co ngay %s -> dung %s",
                    stock, date, row["trade_date"])
    return {
        "stock_symbol": row["stock_symbol"],
        "trade_date": str(row["trade_date"]),
        "close": row["close"],
        "vector": list(row["vector"]),
    }


# ---------------------------------------------------------------------------
# QUERY: An Do qua LSH
# ---------------------------------------------------------------------------

def query_precedents_lsh(
    spark: SparkSession,
    index_dir: str,
    query_vector: list,
    query_buckets: list,
    top_k: int,
) -> list:
    """Tim top_k tien le An Do giong nhat qua SimHash candidate + cosine."""
    df = spark.read.parquet(index_dir)
    candidates = df.filter(
        F.arrays_overlap(col("band_buckets"), F.array(*[lit(b) for b in query_buckets]))
    )
    n_cand = candidates.count()
    log.info("QUERY LSH | %s candidates sau SimHash (truoc khi cosine)", f"{n_cand:,}")
    if n_cand == 0:
        return []
    return _rank_by_cosine(spark, candidates, query_vector, top_k)


# ---------------------------------------------------------------------------
# QUERY: VN cosine exact (kem chong look-ahead)
# ---------------------------------------------------------------------------

def query_precedents_exact(
    spark: SparkSession,
    vectors_dir: str,
    query_vector: list,
    top_k: int,
    query_stock: str = None,
    before_date: str = None,
) -> list:
    """Tim top_k tien le VN giong nhat bang cosine vet can.

    - before_date: chi lay tien le co trade_date < before_date (tranh nhin
      tuong lai khi kiem chung noi dia).
    - query_stock + before_date: loai chinh pattern dang xet.
    """
    df = spark.read.parquet(vectors_dir)
    if before_date is not None:
        df = df.filter(col("trade_date") < lit(before_date))
    return _rank_by_cosine(spark, df, query_vector, top_k)


# ---------------------------------------------------------------------------
# BASE RATE
# ---------------------------------------------------------------------------

def compute_base_rate(spark: SparkSession, vectors_dir: str) -> dict:
    """Ty le nen cua ca thi truong: % pattern co fwd_down + loi suat TB."""
    df = spark.read.parquet(vectors_dir).filter(col("fwd_down").isNotNull())
    row = df.agg(
        F.count("*").alias("n"),
        F.avg("fwd_down").alias("p_down"),
        F.avg("fwd_return").alias("mean_return"),
    ).collect()[0]
    return {
        "n": int(row["n"] or 0),
        "p_down": float(row["p_down"] or 0.0),
        "mean_return": float(row["mean_return"] or 0.0),
    }


# ---------------------------------------------------------------------------
# SUMMARIZE (thong ke top-k + Wilson CI)
# ---------------------------------------------------------------------------

def _wilson_interval(k: int, n: int, z: float):
    """Khoang tin cay Wilson cho ty le k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_precedents(precedents: list, base_rate: dict, ci_z: float) -> dict:
    """Thong ke top-k tien le so voi base rate.

    Returns dict: n, p_down, ci_low, ci_high, mean_return, worst_return,
    excess (p_down - base_rate), avg_cosine.
    """
    n = len(precedents)
    if n == 0:
        return {
            "n": 0, "p_down": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "mean_return": 0.0, "worst_return": 0.0, "excess": 0.0,
            "avg_cosine": 0.0, "base_rate": base_rate.get("p_down", 0.0),
        }

    n_down = sum(int(p["fwd_down"]) for p in precedents)
    returns = [float(p["fwd_return"]) for p in precedents]
    cosines = [float(p["cosine"]) for p in precedents]
    p_down = n_down / n
    ci_low, ci_high = _wilson_interval(n_down, n, ci_z)

    return {
        "n": n,
        "p_down": p_down,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_return": sum(returns) / n,
        "worst_return": min(returns),
        "excess": p_down - base_rate.get("p_down", 0.0),
        "avg_cosine": sum(cosines) / n,
        "base_rate": base_rate.get("p_down", 0.0),
    }
