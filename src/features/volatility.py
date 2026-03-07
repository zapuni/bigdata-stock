"""
Volatility & Anomaly Features
Member C Task: Create volatility-based derived features

Features created:
- bb_width: Bollinger Band Width
- pct_b: Percent B (price location within bands)
- volatility_atr_pct: ATR as percentage of price
- is_high_volatility: Binary high volatility indicator
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, avg


def add_volatility_features(df: DataFrame) -> DataFrame:
    """
    Add volatility and anomaly detection features to the DataFrame.
    
    Features:
    - bb_width: (upperband - lowerband) / middleband
    - pct_b: (Close - lowerband) / (upperband - lowerband)
    - volatility_atr_pct: ATR / Close * 100
    - is_high_volatility: 1 if bb_width > avg(bb_width) * 1.5 per stock
    
    Args:
        df: Input Spark DataFrame with raw technical indicators
        
    Returns:
        DataFrame with added volatility features
    """
    # bb_width: Bollinger Band Width
    # Formula: (upperband - lowerband) / middleband
    # Measures volatility - wider bands = higher volatility
    df = df.withColumn(
        "bb_width",
        when(
            col("middleband").isNotNull() & (col("middleband") != 0) &
            col("upperband").isNotNull() & col("lowerband").isNotNull(),
            (col("upperband") - col("lowerband")) / col("middleband")
        ).otherwise(lit(None))
    )
    
    # pct_b: Percent B (Price location within Bollinger Bands)
    # Formula: (Close - lowerband) / (upperband - lowerband)
    # Values: < 0 = below lower band, > 1 = above upper band
    # 0.5 = at middle band
    df = df.withColumn(
        "pct_b",
        when(
            col("upperband").isNotNull() & col("lowerband").isNotNull() &
            ((col("upperband") - col("lowerband")) != 0),
            (col("close") - col("lowerband")) / (col("upperband") - col("lowerband"))
        ).otherwise(lit(None))
    )
    
    # volatility_atr_pct: ATR as percentage of price
    # Formula: ATR / Close * 100
    # Higher values indicate higher volatility relative to price
    df = df.withColumn(
        "volatility_atr_pct",
        when(
            col("ATR").isNotNull() & col("close").isNotNull() & (col("close") != 0),
            (col("ATR") / col("close")) * 100
        ).otherwise(lit(None))
    )
    
    # is_high_volatility: dung groupBy+join thay vi Window.partitionBy khong gioi han
    # Window.partitionBy("stock_symbol") can load toan bo ~628K rows/stock vao RAM
    # groupBy+join chi can aggregate (O(1) memory per group), an toan hon cho 32GB RAM
    avg_stats = df.groupBy("stock_symbol").agg(
        avg(col("bb_width")).alias("_avg_bb_width")
    )
    df = df.join(avg_stats, on="stock_symbol", how="left")
    df = df.withColumn(
        "is_high_volatility",
        when(
            col("bb_width").isNotNull() & col("_avg_bb_width").isNotNull(),
            when(col("bb_width") > col("_avg_bb_width") * 1.5, lit(1)).otherwise(lit(0))
        ).otherwise(lit(None))
    )
    df = df.drop("_avg_bb_width")

    return df


def get_volatility_feature_columns():
    """
    Get list of volatility feature column names.
    
    Returns:
        List of column names created by add_volatility_features
    """
    return [
        "bb_width",
        "pct_b",
        "volatility_atr_pct",
        "is_high_volatility"
    ]
