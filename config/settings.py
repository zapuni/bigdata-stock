import os
import sys

from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, LongType, TimestampType
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "stock")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

RAW_CSV_PATH = os.path.join(DATA_DIR, "stock-market-data")
PARQUET_PATH = os.path.join(DATA_DIR, "stock-market-data-parquet")
ENGINEERED_PATH = os.path.join(DATA_DIR, "stock-market-data-engineered")
FINAL_PATH = os.path.join(DATA_DIR, "stock-market-data-final")
LSH_PATH = os.path.join(DATA_DIR, "lsh-similarity")
VN_RAW_CSV_PATH = os.path.join(DATA_DIR, "stock-market-data-vn")
VN_FINAL_PATH = os.path.join(DATA_DIR, "stock-market-data-vn-final")

# --- Phương án A: Cảnh báo theo tiền lệ (Similarity Search) ---
# Artifacts: feature vectors (Ấn Độ + VN) + SimHash index (Ấn Độ) + hyperplanes
PRECEDENT_PATH = os.path.join(DATA_DIR, "precedent-alert")
PRECEDENT_REPORTS_DIR = os.path.join(REPORTS_DIR, "precedent-alert")
LSH_BENCHMARK_REPORTS_DIR = os.path.join(REPORTS_DIR, "lsh-benchmark")
PCA_OUTPUT_PATH = os.path.join(DATA_DIR, "pca-clustering")
PCA_REPORTS_DIR = os.path.join(REPORTS_DIR, "pca-clustering")
GRAPH_OUTPUT_PATH = os.path.join(DATA_DIR, "graph-pagerank")
GRAPH_REPORTS_DIR = os.path.join(REPORTS_DIR, "graph-pagerank")

GRAPH_CONFIG = {
    "corr_threshold": 0.5,
    "pagerank_alpha": 0.85,
    "pagerank_max_iter": 200,
    "pagerank_tol": 1e-6,
    "log_return_col": "log_return",
    "date_col": "trade_date",
    "stock_col": "stock_symbol",
    "min_periods": 100,
    "min_stocks_per_day_pct": 0.5,
    "top_n_display": 20,
    "heatmap_top_n": 30,
    "network_top_n": 40,
    "network_edge_min_weight": 0.65,
}

LSH_CONFIG = {
    "window_days": 20,
    "k_shingle": 2,
    "n_hash": 100,
    "n_bands": 20,
    "rows_per_band": 5,
    "benchmark_n_queries": 1000,
    "benchmark_top_k": 10,
    "benchmark_seed": 42,
}

# =========================================================================
# PHƯƠNG ÁN A: CẢNH BÁO THEO TIỀN LỆ (cosine similarity + SimHash LSH)
# =========================================================================
# Ý tưởng: mô tả pattern 20 ngày của 1 (cổ phiếu, ngày) bằng 1 feature vector,
# tìm các tiền lệ GIỐNG NHẤT trong quá khứ (Ấn Độ qua SimHash, VN qua cosine
# exact), thống kê kết quả 3 ngày sau → hợp nhất thành 1 risk score.
PRECEDENT_CONFIG = {
    # --- Pattern & nhãn kết quả ---
    "window_days": 20,          # số ngày giao dịch tạo thành 1 pattern
    "horizon_days": 3,          # nhìn kết quả bao nhiêu ngày sau (next_3d)
    "down_threshold": 0.0,      # fwd_return < ngưỡng này => coi là "giảm"

    # --- Hybrid feature vector ---
    # Phần "hình dạng": chuỗi daily_return 20 ngày, z-score TRONG cửa sổ
    #   (scale-invariant, chỉ giữ hình dạng biến động).
    # Phần "tóm tắt": các đặc trưng dưới, z-score THEO TỪNG THỊ TRƯỜNG.
    "summary_features": [
        "cum_return",     # cộng dồn lợi suất 20 ngày
        "volatility",     # độ lệch chuẩn của daily_return trong cửa sổ
        "volume_ratio",   # volume ngày cuối / volume trung bình cửa sổ
        "rsi",            # RSI14 ngày cuối
        "macd",           # macd1226 ngày cuối (đã chuẩn hoá theo giá)
        "sma_gap",        # (close - sma20) / sma20 ngày cuối
    ],

    # --- SimHash LSH (random hyperplane cho cosine) ---
    # Tổng số hyperplane = band_bits * n_bands. Hai vector là "candidate"
    # nếu trùng HẾT trên ít nhất 1 band → sau đó tính cosine exact để xếp hạng.
    "simhash_band_bits": 10,    # số bit mỗi band (r)
    "simhash_n_bands": 12,      # số band (L) — tăng recall
    "random_seed": 42,

    # --- Truy vấn & thống kê ---
    "top_k": 50,                # số tiền lệ giống nhất lấy ra để thống kê
    "ci_z": 1.96,               # z cho khoảng tin cậy Wilson 95%

    # --- Hợp nhất tín hiệu (Step 7) ---
    # Khi mẫu càng nhiều + khoảng tin cậy càng chặt → trọng số càng lớn.
    # Khi Ấn Độ và VN mâu thuẫn → ưu tiên VN (nhân trọng số cao hơn).
    "vn_priority_weight": 1.5,  # hệ số ưu tiên VN khi hợp nhất
    "risk_caution_threshold": 40,   # >= : THẬN TRỌNG
    "risk_strong_threshold": 65,    # >= : CẢNH BÁO MẠNH
}


PCA_CONFIG = {
    "variance_threshold": 0.90,
    "k_default": 15,
    "kmeans_k": 6,
    "elbow_k_range": (2, 10),
    "cure_n_rep": 10,
    "cure_alpha": 0.2,
    "random_seed": 42,
}


SPARK_CONFIG = {
    "app_name": "StockMarketAnalysis",
    # local[4]: gioi han 4 tasks dong thoi, moi task can ~400MB cho Window buffer
    # local[*]=16 cores => 16*400MB=6.4GB Window + overhead => OOM tren 32GB RAM
    # local[4] => 4*400MB=1.6GB Window, du headroom cho 24g heap
    "master": "local[4]",
    "driver_memory": "24g",
    "executor_memory": "24g",
    "max_result_size": "4g",
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.2",
    "spark.sql.shuffle.partitions": "200",
    "adaptive_enabled": "true",
    "coalesce_partitions_enabled": "true",
    "compression_codec": "snappy",
    "window_spill_threshold": "4096",
}

JAVA_HOME = os.environ.get("JAVA_HOME", "/opt/jdk-17.0.18")
SPARK_HOME = os.environ.get("SPARK_HOME", "/opt/spark-4.0.2-bin-hadoop3")

OHLC_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

MOVING_AVERAGE_COLUMNS = [
    "sma5", "sma10", "sma15", "sma20",
    "ema5", "ema10", "ema15", "ema20"
]

BOLLINGER_COLUMNS = ["upperband", "middleband", "lowerband"]

TREND_COLUMNS = [
    "HT_TRENDLINE", "KAMA10", "KAMA20", "KAMA30",
    "SAR", "TRIMA5", "TRIMA10", "TRIMA20"
]

MOMENTUM_COLUMNS = [
    "ADX5", "ADX10", "ADX20", "APO",
    "CCI5", "CCI10", "CCI15"
]

MACD_COLUMNS = ["macd510", "macd520", "macd1020", "macd1520", "macd1226"]

RATE_OF_CHANGE_COLUMNS = [
    "MFI", "MOM10", "MOM15", "MOM20",
    "ROC5", "ROC10", "ROC20", "PPO"
]

RSI_COLUMNS = ["RSI14", "RSI8"]

STOCHASTIC_COLUMNS = [
    "slowk", "slowd", "fastk", "fastd", "fastksr", "fastdsr"
]

OSCILLATOR_COLUMNS = ["ULTOSC", "WILLR"]

VOLATILITY_COLUMNS = ["ATR", "Trange", "TYPPRICE"]

OTHER_COLUMNS = ["HT_DCPERIOD", "BETA"]

# All feature columns (excluding date and OHLCV)
FEATURE_COLUMNS = (
    MOVING_AVERAGE_COLUMNS +
    BOLLINGER_COLUMNS +
    TREND_COLUMNS +
    MOMENTUM_COLUMNS +
    MACD_COLUMNS +
    RATE_OF_CHANGE_COLUMNS +
    RSI_COLUMNS +
    STOCHASTIC_COLUMNS +
    OSCILLATOR_COLUMNS +
    VOLATILITY_COLUMNS +
    OTHER_COLUMNS
)


PARTITION_COLUMNS = ["year", "month", "stock_symbol"]

FEATURE_GROUPS = {
    "pca_input": [
        "sma5", "sma10", "sma15", "sma20", "ema5", "ema10", "ema15", "ema20",
        "upperband", "middleband", "lowerband", "HT_TRENDLINE",
        "KAMA10", "KAMA20", "KAMA30", "SAR", "TRIMA5", "TRIMA10", "TRIMA20",
        "ADX5", "ADX10", "ADX20", "APO", "CCI5", "CCI10", "CCI15",
        "macd510", "macd520", "macd1020", "macd1520", "macd1226",
        "MOM10", "MOM15", "MOM20", "ROC5", "ROC10", "ROC20", "PPO",
        "RSI14", "RSI8", "slowk", "slowd", "fastk", "fastd", "fastksr", "fastdsr",
        "ULTOSC", "WILLR", "ATR", "Trange", "TYPPRICE", "HT_DCPERIOD", "BETA",
        "dist_sma20", "dist_sma_diff", "bb_width", "pct_b", "volatility_atr_pct",
        "daily_range_pct", "log_return",
    ],
    "lsh_fingerprint": [
        "rsi_status_num", "trend_ema_cross", "stoch_k_d_cross",
        "roc_trend", "is_high_volatility", "macd_cross_signal",
        "bb_position_label", "adx_strength_label",
    ],
    "association_items": [
        "rsi_status", "bb_position_label", "macd_cross_signal",
        "trend_ema_cross", "adx_strength_label", "is_high_volatility",
    ],
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    "max_missing_pct": 20.0,
    "outlier_iqr_multiplier": 1.5,
    "min_completeness_score": 80.0
}

def get_csv_schema():
    """
    Get schema for reading raw CSV files.
    Returns StructType with all 58 feature columns plus OHLCV.
    """
    return StructType([
        # Date
        StructField("date", StringType(), True),
        # OHLCV
        StructField("close", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("open", DoubleType(), True),
        StructField("volume", LongType(), True),
        # Moving Averages
        StructField("sma5", DoubleType(), True),
        StructField("sma10", DoubleType(), True),
        StructField("sma15", DoubleType(), True),
        StructField("sma20", DoubleType(), True),
        StructField("ema5", DoubleType(), True),
        StructField("ema10", DoubleType(), True),
        StructField("ema15", DoubleType(), True),
        StructField("ema20", DoubleType(), True),
        # Bollinger Bands
        StructField("upperband", DoubleType(), True),
        StructField("middleband", DoubleType(), True),
        StructField("lowerband", DoubleType(), True),
        # Trend Indicators
        StructField("HT_TRENDLINE", DoubleType(), True),
        StructField("KAMA10", DoubleType(), True),
        StructField("KAMA20", DoubleType(), True),
        StructField("KAMA30", DoubleType(), True),
        StructField("SAR", DoubleType(), True),
        StructField("TRIMA5", DoubleType(), True),
        StructField("TRIMA10", DoubleType(), True),
        StructField("TRIMA20", DoubleType(), True),
        # ADX
        StructField("ADX5", DoubleType(), True),
        StructField("ADX10", DoubleType(), True),
        StructField("ADX20", DoubleType(), True),
        StructField("APO", DoubleType(), True),
        # CCI
        StructField("CCI5", DoubleType(), True),
        StructField("CCI10", DoubleType(), True),
        StructField("CCI15", DoubleType(), True),
        # MACD
        StructField("macd510", DoubleType(), True),
        StructField("macd520", DoubleType(), True),
        StructField("macd1020", DoubleType(), True),
        StructField("macd1520", DoubleType(), True),
        StructField("macd1226", DoubleType(), True),
        # MFI & Momentum
        StructField("MFI", DoubleType(), True),
        StructField("MOM10", DoubleType(), True),
        StructField("MOM15", DoubleType(), True),
        StructField("MOM20", DoubleType(), True),
        # Rate of Change
        StructField("ROC5", DoubleType(), True),
        StructField("ROC10", DoubleType(), True),
        StructField("ROC20", DoubleType(), True),
        StructField("PPO", DoubleType(), True),
        # RSI
        StructField("RSI14", DoubleType(), True),
        StructField("RSI8", DoubleType(), True),
        # Stochastic
        StructField("slowk", DoubleType(), True),
        StructField("slowd", DoubleType(), True),
        StructField("fastk", DoubleType(), True),
        StructField("fastd", DoubleType(), True),
        StructField("fastksr", DoubleType(), True),
        StructField("fastdsr", DoubleType(), True),
        # Oscillators
        StructField("ULTOSC", DoubleType(), True),
        StructField("WILLR", DoubleType(), True),
        # Volatility
        StructField("ATR", DoubleType(), True),
        StructField("Trange", DoubleType(), True),
        StructField("TYPPRICE", DoubleType(), True),
        # Others
        StructField("HT_DCPERIOD", DoubleType(), True),
        StructField("BETA", DoubleType(), True),
    ])


def get_parquet_schema():
    """
    Get schema for Parquet files (includes metadata columns).
    """
    csv_schema = get_csv_schema()
    metadata_fields = [
        StructField("stock_symbol", StringType(), False),
        StructField("timestamp", TimestampType(), True),
        StructField("year", IntegerType(), False),
        StructField("month", IntegerType(), False),
    ]
    return StructType(csv_schema.fields + metadata_fields)


def get_optimized_schema():
    """
    Get optimized schema with proper data types and grouping.
    Used for schema documentation.
    """
    fields = [
        # Primary Keys
        StructField("stock_symbol", StringType(), False),
        StructField("timestamp", TimestampType(), False),
        # Partition Keys
        StructField("year", IntegerType(), False),
        StructField("month", IntegerType(), False),
        # OHLCV
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", LongType(), True),
        # Moving Averages
        StructField("sma5", DoubleType(), True),
        StructField("sma10", DoubleType(), True),
        StructField("sma15", DoubleType(), True),
        StructField("sma20", DoubleType(), True),
        StructField("ema5", DoubleType(), True),
        StructField("ema10", DoubleType(), True),
        StructField("ema15", DoubleType(), True),
        StructField("ema20", DoubleType(), True),
        # Bollinger Bands
        StructField("upperband", DoubleType(), True),
        StructField("middleband", DoubleType(), True),
        StructField("lowerband", DoubleType(), True),
        # Trend Indicators
        StructField("ht_trendline", DoubleType(), True),
        StructField("kama10", DoubleType(), True),
        StructField("kama20", DoubleType(), True),
        StructField("kama30", DoubleType(), True),
        StructField("sar", DoubleType(), True),
        StructField("trima5", DoubleType(), True),
        StructField("trima10", DoubleType(), True),
        StructField("trima20", DoubleType(), True),
        # Momentum
        StructField("adx5", DoubleType(), True),
        StructField("adx10", DoubleType(), True),
        StructField("adx20", DoubleType(), True),
        StructField("apo", DoubleType(), True),
        StructField("cci5", DoubleType(), True),
        StructField("cci10", DoubleType(), True),
        StructField("cci15", DoubleType(), True),
        # MACD
        StructField("macd510", DoubleType(), True),
        StructField("macd520", DoubleType(), True),
        StructField("macd1020", DoubleType(), True),
        StructField("macd1520", DoubleType(), True),
        StructField("macd1226", DoubleType(), True),
        # Rate of Change
        StructField("mfi", DoubleType(), True),
        StructField("mom10", DoubleType(), True),
        StructField("mom15", DoubleType(), True),
        StructField("mom20", DoubleType(), True),
        StructField("roc5", DoubleType(), True),
        StructField("roc10", DoubleType(), True),
        StructField("roc20", DoubleType(), True),
        StructField("ppo", DoubleType(), True),
        # RSI
        StructField("rsi14", DoubleType(), True),
        StructField("rsi8", DoubleType(), True),
        # Stochastic
        StructField("slowk", DoubleType(), True),
        StructField("slowd", DoubleType(), True),
        StructField("fastk", DoubleType(), True),
        StructField("fastd", DoubleType(), True),
        StructField("fastksr", DoubleType(), True),
        StructField("fastdsr", DoubleType(), True),
        # Oscillators
        StructField("ultosc", DoubleType(), True),
        StructField("willr", DoubleType(), True),
        # Volatility
        StructField("atr", DoubleType(), True),
        StructField("trange", DoubleType(), True),
        StructField("typprice", DoubleType(), True),
        # Others
        StructField("ht_dcperiod", DoubleType(), True),
        StructField("beta", DoubleType(), True),
    ]
    return StructType(fields)