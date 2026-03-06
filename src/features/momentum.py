"""
Momentum & Oscillator Features
Member B Task: Create momentum-based derived features

Features created:
- rsi_status: Categorical RSI interpretation (Overbought/Oversold/Neutral)
- rsi_status_num: Numeric RSI status (1=Overbought, -1=Oversold, 0=Neutral)
- macd_hist_norm: MACD histogram (difference between fast and slow MACD)
- stoch_k_d_cross: Stochastic crossover signal
- roc_trend: Rate of Change trend direction
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit


def add_momentum_features(df: DataFrame) -> DataFrame:
    """
    Add momentum and oscillator features to the DataFrame.
    
    Features:
    - rsi_status: "Overbought" if RSI14 > 70, "Oversold" if < 30, else "Neutral"
    - rsi_status_num: Numeric version (1, -1, 0)
    - macd_hist_norm: macd1226 - macd510 (approximation of histogram)
    - stoch_k_d_cross: 1 if slowk > slowd (bullish), else 0
    - roc_trend: 1 if ROC10 > 0, else -1
    
    Args:
        df: Input Spark DataFrame with raw technical indicators
        
    Returns:
        DataFrame with added momentum features
    """
    # rsi_status: Categorical RSI interpretation
    # RSI > 70 = Overbought, RSI < 30 = Oversold, else Neutral
    df = df.withColumn(
        "rsi_status",
        when(col("RSI14").isNull(), lit(None))
        .when(col("RSI14") > 70, lit("Overbought"))
        .when(col("RSI14") < 30, lit("Oversold"))
        .otherwise(lit("Neutral"))
    )
    
    # rsi_status_num: Numeric version for ML algorithms
    # 1 = Overbought, -1 = Oversold, 0 = Neutral
    df = df.withColumn(
        "rsi_status_num",
        when(col("RSI14").isNull(), lit(None))
        .when(col("RSI14") > 70, lit(1))
        .when(col("RSI14") < 30, lit(-1))
        .otherwise(lit(0))
    )
    
    # macd_hist_norm: MACD histogram approximation
    # Using difference between macd1226 (12,26 period) and macd510 (5,10 period)
    # This represents momentum divergence
    # Note: Without explicit signal line, we use difference between MACD variants
    df = df.withColumn(
        "macd_hist_norm",
        when(
            col("macd1226").isNotNull() & col("macd510").isNotNull(),
            col("macd1226") - col("macd510")
        ).otherwise(lit(None))
    )
    
    # stoch_k_d_cross: Stochastic %K/%D crossover
    # 1 if slowk > slowd (bullish cross), 0 otherwise
    df = df.withColumn(
        "stoch_k_d_cross",
        when(
            col("slowk").isNotNull() & col("slowd").isNotNull(),
            when(col("slowk") > col("slowd"), lit(1)).otherwise(lit(0))
        ).otherwise(lit(None))
    )
    
    # roc_trend: Rate of Change trend direction
    # 1 if ROC10 > 0 (upward momentum), -1 if ROC10 <= 0 (downward momentum)
    df = df.withColumn(
        "roc_trend",
        when(
            col("ROC10").isNotNull(),
            when(col("ROC10") > 0, lit(1)).otherwise(lit(-1))
        ).otherwise(lit(None))
    )
    
    return df


def get_momentum_feature_columns():
    """
    Get list of momentum feature column names.
    
    Returns:
        List of column names created by add_momentum_features
    """
    return [
        "rsi_status",
        "rsi_status_num",
        "macd_hist_norm",
        "stoch_k_d_cross",
        "roc_trend"
    ]
