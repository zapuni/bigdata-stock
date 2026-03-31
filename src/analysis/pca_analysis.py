"""
PCA Analysis Module -- Phase 2: Mining Massive Datasets

Pipeline:
  STAGE 1  compute_stock_profiles : 63.9M rows -> 101 mean feature vectors
  STAGE 2  fit_pca_full            : Fit PCA(k=n_features) -> explained variance
  STAGE 3  find_k_optimal          : Tim k giu >= 90% variance
  STAGE 4  fit_pca_pipeline        : Fit final pipeline (assembler+scaler+PCA)
  STAGE 5  extract_loadings        : Phan tich loadings (PC1=Trend, PC2=Vol...)
  STAGE 6  plot_scree              : Scree + cumulative variance plot

Tai sao PCA truoc Clustering:
- Curse of dimensionality: voi d=60, Euclidean distance giua moi cap diem hoi tu
  ve cung gia tri => K-means khong phan biet duoc cluster
- Multicollinearity: sma5/sma10/ema5/ema10 correlated rho > 0.95
  => 4 cot cung do 1 thong tin "trend" nhung chiem 4/60 = 6.7% chieu
- PCA giu 90%+ variance trong ~15 PCs, khu duoc multicollinearity
"""

import os
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import mean, col, count
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

log = logging.getLogger("stock_pca")


# ---------------------------------------------------------------------------
# STAGE 1: COMPUTE STOCK PROFILES
# ---------------------------------------------------------------------------

def compute_stock_profiles(
    spark: SparkSession,
    final_path: str,
    pca_cols: List[str],
) -> DataFrame:
    """Tinh mean feature vector per stock tu Parquet final.

    63.9M rows (1-min, 101 stocks, 10 nam) -> 101 rows (1 vector / stock).
    Moi stock duoc dai dien boi "chan dung trung binh" trong 10 nam.

    Args:
        spark      : SparkSession
        final_path : duong dan toi ETL final parquet
        pca_cols   : list cot numeric de aggregate (FEATURE_GROUPS["pca_input"])

    Returns:
        Spark DataFrame: stock_symbol, trading_days, <pca_cols>
    """
    log.info("STAGE 1 PROFILES | Reading from %s", final_path)
    df = spark.read.parquet(final_path)

    log.info("STAGE 1 PROFILES | Aggregating %d features over %s rows",
             len(pca_cols), f"{df.count():,}")

    # Bo nhung row null log_return (warm-up dau period)
    df = df.filter(col("log_return").isNotNull())

    agg_exprs = [mean(c).alias(c) for c in pca_cols]
    agg_exprs.append(count("*").alias("trading_days"))

    profiles = (
        df.groupBy("stock_symbol")
        .agg(*agg_exprs)
        .orderBy("stock_symbol")
    )

    n_stocks = profiles.count()
    log.info("STAGE 1 PROFILES | Done: %d stocks x %d features", n_stocks, len(pca_cols))
    return profiles


# ---------------------------------------------------------------------------
# STAGE 2-3: PCA FULL FIT + FIND K OPTIMAL
# ---------------------------------------------------------------------------

def _build_assembler_scaler(pca_cols: List[str]) -> Tuple[VectorAssembler, StandardScaler]:
    """Helper: tao VectorAssembler + StandardScaler reuse cho ca full va final fit."""
    assembler = VectorAssembler(
        inputCols=pca_cols,
        outputCol="features_raw",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=True,
        withStd=True,
    )
    return assembler, scaler


def fit_pca_full(
    profiles: DataFrame,
    pca_cols: List[str],
) -> Tuple[PipelineModel, np.ndarray, np.ndarray]:
    """Fit PCA voi k=n_features de lay full explained variance array.

    Returns:
        prep_model   : PipelineModel da fit (assembler + scaler) - reuse cho final
        ev_array     : np.array(n_features) explained variance per PC
        cumsum_array : np.array(n_features) cumulative explained variance
    """
    n_features = len(pca_cols)
    n_samples = profiles.count()
    k_max = min(n_features, n_samples - 1)

    log.info("STAGE 2 PCA FULL | Fitting k_max=%d on %dx%d", k_max, n_samples, n_features)

    assembler, scaler = _build_assembler_scaler(pca_cols)
    pca_full = PCA(k=k_max, inputCol="features_scaled", outputCol="pca_full")
    pipeline = Pipeline(stages=[assembler, scaler, pca_full])
    model = pipeline.fit(profiles)

    pca_stage: PCA = model.stages[2]
    ev_array = np.array(pca_stage.explainedVariance.toArray())
    cumsum_array = np.cumsum(ev_array)

    log.info(
        "STAGE 2 PCA FULL | EV[0..4]=%s | cumulative[k=15]=%.3f",
        np.round(ev_array[:5], 4).tolist(),
        cumsum_array[14] if len(cumsum_array) > 14 else cumsum_array[-1],
    )
    return model, ev_array, cumsum_array


def find_k_optimal(cumsum_array: np.ndarray, threshold: float = 0.90) -> int:
    """Tim k nho nhat sao cho cumulative explained variance >= threshold."""
    above = np.where(cumsum_array >= threshold)[0]
    if len(above) == 0:
        log.warning("FIND K | Threshold %.2f not reached, using k_max=%d",
                    threshold, len(cumsum_array))
        return len(cumsum_array)
    k_optimal = int(above[0]) + 1
    log.info("FIND K | k_optimal=%d giu %.2f%% variance (threshold=%.0f%%)",
             k_optimal, cumsum_array[k_optimal - 1] * 100, threshold * 100)
    return k_optimal


# ---------------------------------------------------------------------------
# STAGE 4: FIT FINAL PCA PIPELINE
# ---------------------------------------------------------------------------

def fit_pca_pipeline(
    profiles: DataFrame,
    pca_cols: List[str],
    k: int,
) -> Tuple[PipelineModel, DataFrame]:
    """Fit final pipeline (assembler + scaler + PCA(k)) va transform.

    Args:
        profiles : Spark DataFrame stock profiles
        pca_cols : list feature columns
        k        : so principal components

    Returns:
        model        : PipelineModel da fit (save lai duoc)
        transformed  : DataFrame voi cot pca_features (DenseVector size k)
    """
    log.info("STAGE 4 PCA FINAL | Fitting k=%d", k)
    assembler, scaler = _build_assembler_scaler(pca_cols)
    pca = PCA(k=k, inputCol="features_scaled", outputCol="pca_features")
    pipeline = Pipeline(stages=[assembler, scaler, pca])

    model = pipeline.fit(profiles)
    transformed = model.transform(profiles)

    pca_stage: PCA = model.stages[2]
    ev = np.array(pca_stage.explainedVariance.toArray())
    log.info("STAGE 4 PCA FINAL | k=%d giu %.2f%% variance", k, ev.sum() * 100)
    return model, transformed


# ---------------------------------------------------------------------------
# STAGE 5: EXTRACT LOADINGS
# ---------------------------------------------------------------------------

def extract_loadings(
    pipeline_model: PipelineModel,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Extract loadings matrix tu PCA model thanh pandas DataFrame.

    Loadings cho biet moi PC la to hop tuyen tinh cua features nao.
    Loading lon (positive/negative) = feature dong gop nhieu cho PC do.

    Returns:
        DataFrame index=feature_cols, columns=[PC1, PC2, ..., PCk]
    """
    pca_stage: PCA = pipeline_model.stages[2]
    pc_matrix = np.array(pca_stage.pc.toArray())
    k = pc_matrix.shape[1]

    loadings_df = pd.DataFrame(
        pc_matrix,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(k)],
    )
    log.info("STAGE 5 LOADINGS | Shape: %d features x %d PCs",
             pc_matrix.shape[0], pc_matrix.shape[1])
    return loadings_df


def interpret_top_loadings(
    loadings_df: pd.DataFrame,
    top_n: int = 5,
    n_pcs: int = 5,
) -> str:
    """Format top-N loadings cho top-N PCs thanh text de log/save.

    Output mau:
        PC1 (Trend Factor):
          +0.182 sma5
          +0.179 ema5
          +0.178 sma10
          ...
    """
    lines = []
    for pc_idx in range(min(n_pcs, loadings_df.shape[1])):
        pc_name = f"PC{pc_idx + 1}"
        col_vals = loadings_df[pc_name]
        top = col_vals.abs().nlargest(top_n)
        lines.append(f"\n{pc_name} top {top_n} features (by |loading|):")
        for feat in top.index:
            val = col_vals[feat]
            sign = "+" if val >= 0 else "-"
            lines.append(f"  {sign}{abs(val):.4f}  {feat}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STAGE 6: PLOTS
# ---------------------------------------------------------------------------

def plot_scree(
    ev_array: np.ndarray,
    cumsum_array: np.ndarray,
    k_optimal: int,
    output_path: str,
    threshold: float = 0.90,
    n_show: int = 20,
) -> None:
    """Ve scree plot (bar EV) + cumulative variance plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_show, len(ev_array))
    x = np.arange(1, n_show + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, ev_array[:n_show], color="#377eb8", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot - Variance per PC")
    axes[0].set_xticks(x)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(x, cumsum_array[:n_show], marker="o", linewidth=2, color="#e41a1c")
    axes[1].axhline(y=threshold, color="green", linestyle="--",
                    label=f"{int(threshold*100)}% threshold")
    axes[1].axvline(x=k_optimal, color="orange", linestyle="--",
                    label=f"k_optimal = {k_optimal}")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Variance vs k")
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT SCREE | Saved -> %s", output_path)


# ---------------------------------------------------------------------------
# UTILITY: Extract PC scores tu DenseVector thanh scalar columns
# ---------------------------------------------------------------------------

def explode_pc_columns(df: DataFrame, k: int, vec_col: str = "pca_features") -> DataFrame:
    """Tach DenseVector pca_features thanh k cot pc1, pc2, ..., pck.

    Useful cho clustering/visualization khi can scalar columns.
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType

    for i in range(k):
        idx = i

        def _make_extractor(j):
            return udf(lambda v: float(v[j]) if v is not None else None, FloatType())

        df = df.withColumn(f"pc{i + 1}", _make_extractor(idx)(vec_col))
    return df
