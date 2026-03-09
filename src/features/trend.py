
"""
Trend Analysis Features
Member A Task: Create trend-based derived features

Features created:
- dist_sma20: Percentage distance from Close to SMA20
- dist_sma_diff: Difference between SMA20 and SMA10 (proxy for longer trend)
- trend_ema_cross: Binary signal for EMA5 > EMA20 crossover
- trend_strength_adx: Normalized ADX strength indicator
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit


def add_trend_features(df: DataFrame) -> DataFrame:
    """
    Add trend analysis features to the DataFrame.
    
    Features:
    - dist_sma20: (Close - SMA20) / SMA20 * 100
    - dist_sma_diff: (SMA20 - SMA10) / SMA10 * 100 (trend direction proxy)
    - trend_ema_cross: 1 if EMA5 > EMA20, else -1
    - trend_strength_adx: 1 if ADX > 25 (Strong), else 0 (Weak)
    
    Args:
        df: Input Spark DataFrame with raw technical indicators
        
    Returns:
        DataFrame with added trend features
    """
    # dist_sma20: Percentage distance from Close to SMA20
    # Formula: (Close - SMA20) / SMA20 * 100
    df = df.withColumn(
        "dist_sma20",
        when(
            col("sma20").isNotNull() & (col("sma20") != 0),
            ((col("close") - col("sma20")) / col("sma20")) * 100
        ).otherwise(lit(None))
    )
    
    # dist_sma_diff: SMA difference as proxy for longer-term trend
    # Since sma50 doesn't exist, use (SMA20 - SMA10) / SMA10 * 100
    # Positive = uptrend, Negative = downtrend
    df = df.withColumn(
        "dist_sma_diff",
        when(
            col("sma10").isNotNull() & (col("sma10") != 0),
            ((col("sma20") - col("sma10")) / col("sma10")) * 100
        ).otherwise(lit(None))
    )
    
    # trend_ema_cross: EMA crossover signal
    # 1 if EMA5 > EMA20 (bullish), -1 if EMA5 <= EMA20 (bearish)
    df = df.withColumn(
        "trend_ema_cross",
        when(
            col("ema5").isNotNull() & col("ema20").isNotNull(),
            when(col("ema5") > col("ema20"), lit(1)).otherwise(lit(-1))
        ).otherwise(lit(None))
    )
    
    # trend_strength_adx: ADX-based trend strength
    # Use ADX10 as representative ADX value
    # ADX > 25 = Strong trend (1), ADX <= 25 = Weak trend (0)
    df = df.withColumn(
        "trend_strength_adx",
        when(
            col("ADX10").isNotNull(),
            when(col("ADX10") > 25, lit(1)).otherwise(lit(0))
        ).otherwise(lit(None))
    )
    
    return df


def get_trend_feature_columns():
    """
    Get list of trend feature column names.
    
    Returns:
        List of column names created by add_trend_features
    """
    return [
        "dist_sma20",
        "dist_sma_diff",
        "trend_ema_cross",
        "trend_strength_adx"
    ]
