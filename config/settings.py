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


SPARK_CONFIG = {
    "app_name": "StockMarketAnalysis",
    "master": "local[*]",
    "driver_memory": "8g",
    "executor_memory": "18g",
    "max_result_size": "4g",
    "spark.memory.fraction": "0.8",
    "spark.memory.storageFraction": "0.3",
    "spark.sql.shuffle.partitions": "32",
    "adaptive_enabled": "true",
    "coalesce_partitions_enabled": "true",
    "compression_codec": "snappy"
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