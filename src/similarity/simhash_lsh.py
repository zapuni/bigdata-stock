"""
STEP 3 — SimHash LSH cho cosine (random hyperplane).

LSH chỉ là công cụ TĂNG TỐC tìm kiếm; bản chất vẫn là đo cosine.

Nguyên lý SimHash:
  - Sinh R = band_bits × n_bands siêu phẳng ngẫu nhiên (vector Gaussian dim D).
  - Với vector v: bit_i = 1 nếu dot(v, h_i) ≥ 0, ngược lại 0.
  - Xác suất hai vector trùng 1 bit = 1 − θ/π (θ = góc giữa chúng) → hai vector
    càng giống (cosine cao) càng dễ trùng nhiều bit.
  - Banding: chia R bit thành n_bands band (mỗi band band_bits bit). Hai vector
    là "candidate" nếu trùng HẾT trên ít nhất 1 band. Sau đó tính cosine exact
    để xếp hạng chính xác.

Ấn Độ (dữ liệu lớn) → dùng SimHash để lọc candidate.
VN (dữ liệu nhỏ)    → bỏ qua LSH, tính cosine exact (xem precedent_query.py).

Hyperplanes được lưu ra đĩa và DÙNG CHUNG cho cả Ấn Độ lẫn VN, đảm bảo hai
thị trường được chiếu lên cùng một hệ toạ độ hash.
"""

import os
import time
import logging

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

log = logging.getLogger("precedent")

_HYPERPLANE_FILE = "hyperplanes.npy"


# ---------------------------------------------------------------------------
# HYPERPLANES (sinh 1 lan, dung chung)
# ---------------------------------------------------------------------------

def make_hyperplanes(dim: int, n_bits: int, seed: int) -> np.ndarray:
    """Sinh ma tran sieu phang (n_bits x dim) Gaussian, co dinh seed."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_bits, dim)


def save_hyperplanes(planes: np.ndarray, output_path: str) -> str:
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, _HYPERPLANE_FILE)
    np.save(path, planes)
    return path


def load_hyperplanes(output_path: str) -> np.ndarray:
    path = os.path.join(output_path, _HYPERPLANE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay hyperplanes tai {path}. Hay build index truoc."
        )
    return np.load(path)


# ---------------------------------------------------------------------------
# SIMHASHER (driver-side, dung cho query 1 vector)
# ---------------------------------------------------------------------------

class SimHasher:
    """Tinh band buckets cho 1 vector tu ma tran hyperplane da co."""

    def __init__(self, planes: np.ndarray, band_bits: int, n_bands: int):
        assert planes.shape[0] == band_bits * n_bands, (
            f"hyperplanes {planes.shape[0]} != band_bits*n_bands "
            f"{band_bits * n_bands}"
        )
        self.planes = planes
        self.band_bits = band_bits
        self.n_bands = n_bands

    def buckets(self, vector) -> list:
        """Tra ve list[str] cac bucket key (1 key / band)."""
        v = np.asarray(vector, dtype=float)
        bits = (self.planes @ v >= 0).astype(int)  # (R,)
        return _bits_to_buckets(bits.tolist(), self.band_bits, self.n_bands)


def _bits_to_buckets(bits: list, band_bits: int, n_bands: int) -> list:
    out = []
    for b in range(n_bands):
        chunk = bits[b * band_bits : (b + 1) * band_bits]
        out.append(f"{b}:" + "".join(str(int(x)) for x in chunk))
    return out


# ---------------------------------------------------------------------------
# BAND BUCKET UDF (dung chung: build index An Do + gan buckets cho VN backtest)
# ---------------------------------------------------------------------------

def make_bucket_udf(spark: SparkSession, planes, band_bits: int, n_bands: int):
    """Tao UDF tinh band buckets tu hyperplanes (broadcast cho workers)."""
    planes_bc = spark.sparkContext.broadcast(
        planes.tolist() if hasattr(planes, "tolist") else planes
    )

    @F.udf(returnType=ArrayType(StringType()))
    def _bucket_udf(vector):
        if not vector:
            return []
        pl = planes_bc.value
        bits = []
        for h in pl:
            s = 0.0
            for a, b in zip(h, vector):
                s += a * b
            bits.append(1 if s >= 0 else 0)
        return _bits_to_buckets(bits, band_bits, n_bands)

    return _bucket_udf


def add_band_buckets(spark: SparkSession, df, planes, band_bits: int, n_bands: int,
                     vector_col: str = "vector", out_col: str = "band_buckets"):
    """Them cot band_buckets vao df dua tren hyperplanes da co."""
    bucket_udf = make_bucket_udf(spark, planes, band_bits, n_bands)
    return df.withColumn(out_col, bucket_udf(vector_col))


# ---------------------------------------------------------------------------
# BUILD INDEX (Spark, cho An Do)
# ---------------------------------------------------------------------------

def build_simhash_index(
    spark: SparkSession,
    vectors_dir: str,
    output_path: str,
    config: dict,
) -> int:
    """Doc vectors An Do -> gan band buckets -> luu index parquet.

    Args:
        spark: SparkSession.
        vectors_dir: thu muc vectors cua An Do (build_feature_vectors output).
        output_path: thu muc PRECEDENT_PATH (luu hyperplanes + simhash-index).
        config: PRECEDENT_CONFIG.

    Returns:
        So pattern da index.
    """
    t0 = time.time()
    band_bits = config["simhash_band_bits"]
    n_bands = config["simhash_n_bands"]
    n_bits = band_bits * n_bands
    seed = config["random_seed"]

    df = spark.read.parquet(vectors_dir)
    dim = len(df.select("vector").first()["vector"])

    planes = make_hyperplanes(dim, n_bits, seed)
    save_hyperplanes(planes, output_path)
    log.info("SIMHASH | dim=%d, R=%d bits (%d bands x %d), seed=%d",
             dim, n_bits, n_bands, band_bits, seed)

    indexed = add_band_buckets(spark, df, planes, band_bits, n_bands)

    index_dir = os.path.join(output_path, "simhash-index")
    (
        indexed.select(
            "stock_symbol", "trade_date", "close",
            "fwd_return", "fwd_down", "vector", "band_buckets",
        )
        .write.mode("overwrite").parquet(index_dir)
    )

    total = indexed.count()
    log.info("SIMHASH | indexed %s patterns in %.1fs -> %s",
             f"{total:,}", time.time() - t0, index_dir)
    return total
