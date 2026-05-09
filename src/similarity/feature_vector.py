"""
STEP 2 — Tạo feature vector cho mỗi pattern (dùng chung Ấn Độ + VN).

Mỗi pattern = 20 ngày giao dịch gần nhất của 1 (cổ phiếu, ngày). Ta mô tả nó
bằng một HYBRID feature vector:

    [ z-score(daily_return) × 20 ]  ++  [ 6 đặc trưng tóm tắt z-score/market ]
      \___ phần "hình dạng" ___/         \____ phần "chế độ thị trường" ____/

- Phần hình dạng: chuỗi lợi suất 20 ngày, z-score TRONG cửa sổ → chỉ giữ
  hình dạng biến động, bỏ qua scale (giá VN và Ấn Độ khác cỡ nhau). Đây là
  phần giúp cosine so sánh "lên/xuống/rung lắc thế nào" một cách công bằng.
- Phần tóm tắt: cum_return, volatility, volume_ratio, RSI, MACD, SMA_gap ở
  ngày cuối cửa sổ → z-score theo TỪNG THỊ TRƯỜNG (mean/std của chính
  thị trường đó) để các đặc trưng về cùng thang đo.

Sau cùng vector được L2-normalize ⇒ cosine(u, v) = dot(u, v).

Input : final parquet (output của main_pipeline.py) của 1 thị trường.
Output: parquet với cột `vector` (array<double>) + nhãn kết quả 3 ngày sau.
"""

import os
import json
import time
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lag, when, lit, to_date
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.window import Window

log = logging.getLogger("precedent")


# ---------------------------------------------------------------------------
# STAGE 1: DAILY AGGREGATION (minute -> 1 snapshot/ngay)
# ---------------------------------------------------------------------------

def _daily_snapshot(spark: SparkSession, final_path: str) -> DataFrame:
    """Gop du lieu phut -> 1 dong/ngay cho moi co phieu.

    - close, sma20, RSI14, macd1226: lay gia tri CUOI ngay (last).
    - volume: TONG khoi luong trong ngay (sum) -> dung cho volume_ratio.
    """
    df = spark.read.parquet(final_path)
    df = df.withColumn("trade_date", to_date(col("timestamp")))

    daily = (
        df.groupBy("stock_symbol", "year", "trade_date")
        .agg(
            F.last("close", ignorenulls=True).alias("close"),
            F.last("sma20", ignorenulls=True).alias("sma20"),
            F.last("RSI14", ignorenulls=True).alias("rsi"),
            F.last("macd1226", ignorenulls=True).alias("macd"),
            F.sum("volume").alias("volume"),
        )
        .filter(col("close").isNotNull() & (col("close") > 0))
    )
    return daily


# ---------------------------------------------------------------------------
# STAGE 2: DAILY RETURN + FORWARD OUTCOME (nhan next_3d)
# ---------------------------------------------------------------------------

def _add_return_and_outcome(daily: DataFrame, horizon: int, down_threshold: float) -> DataFrame:
    """Them daily_return (log) va ket qua `horizon` ngay sau.

    fwd_return = close_{t+h} / close_t - 1
    fwd_down   = 1 neu fwd_return < down_threshold (mac dinh 0) else 0
    """
    w = Window.partitionBy("stock_symbol").orderBy("trade_date")
    prev_close = lag("close", 1).over(w)
    fwd_close = lag("close", -horizon).over(w)

    daily = daily.withColumn(
        "daily_return",
        when(prev_close.isNotNull() & (prev_close > 0), F.log(col("close") / prev_close))
        .otherwise(lit(None)),
    )
    daily = daily.withColumn(
        "fwd_return",
        when(fwd_close.isNotNull(), fwd_close / col("close") - 1.0).otherwise(lit(None)),
    )
    daily = daily.withColumn(
        "fwd_down",
        when(col("fwd_return").isNull(), lit(None))
        .when(col("fwd_return") < lit(down_threshold), lit(1))
        .otherwise(lit(0)),
    )
    return daily


# ---------------------------------------------------------------------------
# STAGE 3: ROLLING 20-DAY WINDOW -> raw feature columns
# ---------------------------------------------------------------------------

def _add_window_features(daily: DataFrame, window_days: int) -> DataFrame:
    """Gom `window_days` ngay gan nhat thanh cac cot tho phuc vu vector hoa."""
    w_ret = (
        Window.partitionBy("stock_symbol")
        .orderBy("trade_date")
        .rowsBetween(-window_days + 1, 0)
    )

    daily = daily.withColumn("ret_window", F.collect_list("daily_return").over(w_ret))
    daily = daily.withColumn("vol_window", F.collect_list("volume").over(w_ret))

    # Chi giu cac dong du `window_days` ngay (daily_return ngay dau la null nen
    # collect_list bo qua -> du window can window_days+1 ngay lich su).
    daily = daily.filter(F.size("ret_window") == window_days)

    # --- Cac dac trung tom tat (raw, chua z-score theo thi truong) ---
    daily = daily.withColumn(
        "cum_return", F.aggregate("ret_window", lit(0.0), lambda acc, x: acc + x)
    )
    daily = daily.withColumn("volatility", _std_udf("ret_window"))
    daily = daily.withColumn(
        "volume_ratio",
        when(
            _mean_udf("vol_window") > 0,
            F.element_at("vol_window", -1) / _mean_udf("vol_window"),
        ).otherwise(lit(1.0)),
    )
    daily = daily.withColumn(
        "sma_gap",
        when(col("sma20") > 0, (col("close") - col("sma20")) / col("sma20")).otherwise(lit(0.0)),
    )
    # rsi, macd da co tu daily snapshot.
    daily = daily.withColumn("zret_window", _zscore_seq_udf("ret_window"))
    return daily


# ---------------------------------------------------------------------------
# UDFs (numpy-free, thuan Python -> tranh phu thuoc worker env)
# ---------------------------------------------------------------------------

@F.udf(returnType=DoubleType())
def _std_udf(arr):
    if not arr:
        return 0.0
    vals = [float(x) for x in arr if x is not None]
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return float(var ** 0.5)


@F.udf(returnType=DoubleType())
def _mean_udf(arr):
    if not arr:
        return 0.0
    vals = [float(x) for x in arr if x is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


@F.udf(returnType=ArrayType(DoubleType()))
def _zscore_seq_udf(arr):
    """Z-score chuoi TRONG cua so -> scale-invariant 'hinh dang'."""
    if not arr:
        return []
    vals = [float(x) if x is not None else 0.0 for x in arr]
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    sd = var ** 0.5
    if sd <= 1e-12:
        return [0.0] * n
    return [(v - m) / sd for v in vals]


@F.udf(returnType=ArrayType(DoubleType()))
def _assemble_and_normalize_udf(zret, summary):
    """Noi [zret(20)] ++ [6 summary z] roi L2-normalize toan vector."""
    if not zret or summary is None:
        return []
    vec = [float(x) for x in zret] + [float(x) for x in summary]
    norm = sum(v * v for v in vec) ** 0.5
    if norm <= 1e-12:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# STAGE 4: STANDARDIZE SUMMARY/MARKET + ASSEMBLE + L2-NORMALIZE
# ---------------------------------------------------------------------------

_SUMMARY_RAW_COLS = ["cum_return", "volatility", "volume_ratio", "rsi", "macd", "sma_gap"]


def _standardize_and_assemble(daily: DataFrame):
    """Z-score 6 dac trung tom tat theo thi truong, roi rap thanh vector chuan hoa.

    Returns:
        (df_with_vector, stats_dict) -- stats_dict luu mean/std de query tai lap.
    """
    agg_exprs = []
    for c in _SUMMARY_RAW_COLS:
        agg_exprs += [F.mean(c).alias(f"{c}_m"), F.stddev(c).alias(f"{c}_s")]
    stats_row = daily.agg(*agg_exprs).collect()[0].asDict()

    stats = {}
    for c in _SUMMARY_RAW_COLS:
        m = stats_row[f"{c}_m"] or 0.0
        s = stats_row[f"{c}_s"] or 0.0
        stats[c] = {"mean": float(m), "std": float(s) if s and s > 1e-12 else 1.0}

    z_cols = []
    for c in _SUMMARY_RAW_COLS:
        zc = f"{c}_z"
        daily = daily.withColumn(
            zc, (col(c) - lit(stats[c]["mean"])) / lit(stats[c]["std"])
        )
        z_cols.append(col(zc))

    daily = daily.withColumn("summary_z", F.array(*z_cols))
    daily = daily.withColumn(
        "vector", _assemble_and_normalize_udf(col("zret_window"), col("summary_z"))
    )
    return daily, stats


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def build_feature_vectors(
    spark: SparkSession,
    final_path: str,
    output_path: str,
    market: str,
    config: dict,
) -> int:
    """Build + luu feature vectors cho 1 thi truong.

    Args:
        spark: SparkSession.
        final_path: parquet output cua ETL (main_pipeline.py).
        output_path: thu muc luu vectors cua thi truong nay.
        market: "india" hoac "vn" (chi dung de log).
        config: PRECEDENT_CONFIG.

    Returns:
        So pattern (dong) da tao.
    """
    t0 = time.time()
    window_days = config["window_days"]
    horizon = config["horizon_days"]
    down_threshold = config["down_threshold"]

    log.info("VECTORS [%s] | window=%dd horizon=%dd | reading %s",
             market, window_days, horizon, final_path)

    daily = _daily_snapshot(spark, final_path)
    daily = _add_return_and_outcome(daily, horizon, down_threshold)
    daily = _add_window_features(daily, window_days)
    daily, stats = _standardize_and_assemble(daily)

    out = (
        daily.select(
            "stock_symbol", "trade_date", "year", "close",
            "fwd_return", "fwd_down", "vector",
        )
        .filter(F.size("vector") == window_days + len(_SUMMARY_RAW_COLS))
    )

    os.makedirs(output_path, exist_ok=True)
    vectors_dir = os.path.join(output_path, "vectors")
    out.write.mode("overwrite").partitionBy("year").parquet(vectors_dir)

    # Luu stats chuan hoa -> query co the tai lap vector cho ngay ngoai bang.
    stats_path = os.path.join(output_path, "standardize_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {"market": market, "window_days": window_days,
             "summary_features": _SUMMARY_RAW_COLS, "stats": stats},
            f, indent=2,
        )

    total = out.count()
    log.info("VECTORS [%s] | %s patterns (dim=%d) in %.1fs -> %s",
             market, f"{total:,}", window_days + len(_SUMMARY_RAW_COLS),
             time.time() - t0, vectors_dir)
    return total
