"""
Phase 2 Preparation Features

Tao them cac features phuc vu GD2: PCA, LSH, Clustering, Association Rules.
Input : DataFrame voi ~77 cot tu GD1 feature engineering
Output: DataFrame them ~7 cot chuan bi cho GD2
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lag, log, when, lit, stddev, mean,
)
from pyspark.sql.window import Window


def add_phase2_features(df: DataFrame) -> DataFrame:
    # 1) Gop tat ca lag operations vao 1 buoc: cung Window spec
    #    => Spark Catalyst gop lai thanh 1 pass duy nhat qua data
    df = _add_all_lag_features(df)
    # 2) groupBy+join: khong dung Window => khong can buffer partition
    df = _add_price_zscore(df)
    # 3) Cac column expressions don gian: khong shuffle, khong Window
    df = _add_bb_position_label(df)
    df = _add_adx_strength_label(df)
    df = _add_daily_range_pct(df)
    return df


def _add_all_lag_features(df: DataFrame) -> DataFrame:
    """Tinh tat ca lag-based features trong 1 Window pass duy nhat.

    3 features deu dung cung Window spec: partitionBy(stock_symbol).orderBy(timestamp).
    Gop lai de Spark khong can buffer partition nhieu lan.
      - log_return:        log(close_t / close_{t-1})
      - next_3d_label:     UP/DOWN/FLAT dua tren close_{t+3}
      - macd_cross_signal: BULLISH/BEARISH/NONE dua tren macd1226_{t-1}
    """
    w = Window.partitionBy("stock_symbol").orderBy("timestamp")
    prev_close = lag("close", 1).over(w)
    future_close = lag("close", -3).over(w)
    prev_macd = lag("macd1226", 1).over(w)

    df = df.withColumn(
        "log_return",
        when(
            prev_close.isNotNull() & (prev_close > 0),
            log(col("close") / prev_close),
        ).otherwise(lit(None)),
    )

    df = df.withColumn(
        "next_3d_label",
        when(future_close.isNull(), lit(None))
        .when(future_close > col("close") * 1.01, lit("UP"))
        .when(future_close < col("close") * 0.99, lit("DOWN"))
        .otherwise(lit("FLAT")),
    )

    df = df.withColumn(
        "macd_cross_signal",
        when(prev_macd.isNull(), lit("NONE"))
        .when((prev_macd < 0) & (col("macd1226") >= 0), lit("BULLISH_CROSS"))
        .when((prev_macd > 0) & (col("macd1226") <= 0), lit("BEARISH_CROSS"))
        .otherwise(lit("NONE")),
    )

    return df


def _add_price_zscore(df: DataFrame) -> DataFrame:
    """Z-score cua Close theo tung stock (toan lich su).
    Dung groupBy+join (101 rows) thay vi Window de tranh OOM.
    """
    stats = df.groupBy("stock_symbol").agg(
        mean("close").alias("_avg_c"),
        stddev("close").alias("_std_c"),
    )
    df = df.join(stats.hint("broadcast"), on="stock_symbol", how="left")
    df = df.withColumn(
        "close_zscore",
        when(col("_std_c") > 0, (col("close") - col("_avg_c")) / col("_std_c"))
        .otherwise(lit(0.0)),
    )
    return df.drop("_avg_c", "_std_c")


def _add_bb_position_label(df: DataFrame) -> DataFrame:
    """Vi tri gia trong Bollinger Bands."""
    return df.withColumn(
        "bb_position_label",
        when(col("close") > col("upperband"), lit("ABOVE_UPPER"))
        .when(col("close") > col("middleband"), lit("UPPER_ZONE"))
        .when(col("close") == col("middleband"), lit("AT_MIDDLE"))
        .when(col("close") > col("lowerband"), lit("LOWER_ZONE"))
        .otherwise(lit("BELOW_LOWER")),
    )


def _add_adx_strength_label(df: DataFrame) -> DataFrame:
    """Suc manh xu huong theo ADX20."""
    return df.withColumn(
        "adx_strength_label",
        when(col("ADX20") >= 25, lit("STRONG_TREND"))
        .when(col("ADX20") >= 15, lit("WEAK_TREND"))
        .otherwise(lit("NO_TREND")),
    )


def _add_daily_range_pct(df: DataFrame) -> DataFrame:
    """Bien do giao dong trong ngay theo % = (High - Low) / Close * 100."""
    return df.withColumn(
        "daily_range_pct",
        when(
            col("close") > 0,
            (col("high") - col("low")) / col("close") * 100,
        ).otherwise(lit(None)),
    )


def get_phase2_feature_columns():
    return [
        "log_return",
        "next_3d_label",
        "close_zscore",
        "macd_cross_signal",
        "bb_position_label",
        "adx_strength_label",
        "daily_range_pct",
    ]
