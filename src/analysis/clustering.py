"""
Clustering Module -- Phase 2: Mining Massive Datasets

Pipeline:
  STAGE 1  silhouette_baseline_raw : Silhouette truoc PCA (60 features)
  STAGE 2  kmeans_elbow             : Elbow method tim k toi uu
  STAGE 3  kmeans_fit               : K-means k=6 tren PC space
  STAGE 4  cure_clustering          : CURE algorithm (xu ly outlier)
  STAGE 5  compare_methods          : K-means vs CURE
  STAGE 6  plots                    : Elbow / scatter / cluster profile

Tai sao K-means + CURE:
- K-means: nhanh O(nkd), Spark MLlib native, baseline tot
- CURE: multiple representatives + shrink alpha => bat duoc cluster phi cau,
        khong bi keo boi outlier (ma penny/speculative nhu ADANIPORTS)
- So sanh 2 phuong phap => stocks dong y giua 2 thuat toan = stable assignment
"""

import os
import logging
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator

log = logging.getLogger("stock_pca")


# ---------------------------------------------------------------------------
# STAGE 1: SILHOUETTE BASELINE (TRUOC PCA)
# ---------------------------------------------------------------------------

def silhouette_baseline_raw(
    profiles_scaled: DataFrame,
    k: int,
    seed: int = 42,
) -> Tuple[float, KMeansModel]:
    """K-means + Silhouette tren features_scaled (truoc PCA).

    Dung de chung minh PCA cai thien clustering quality.
    """
    log.info("STAGE 1 BASELINE | K-means k=%d on raw features_scaled", k)
    km = KMeans(
        k=k, seed=seed,
        featuresCol="features_scaled",
        predictionCol="cluster_id_raw",
        maxIter=100, tol=1e-4,
    )
    model = km.fit(profiles_scaled)
    preds = model.transform(profiles_scaled)

    evaluator = ClusteringEvaluator(
        featuresCol="features_scaled",
        predictionCol="cluster_id_raw",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    sil = float(evaluator.evaluate(preds))
    log.info("STAGE 1 BASELINE | Silhouette (BEFORE PCA): %.4f", sil)
    return sil, model


# ---------------------------------------------------------------------------
# STAGE 2: ELBOW METHOD
# ---------------------------------------------------------------------------

def kmeans_elbow(
    stock_pca: DataFrame,
    k_range: Iterable[int],
    seed: int = 42,
) -> List[Tuple[int, float]]:
    """Chay K-means voi nhieu k de plot elbow curve.

    Returns:
        list[(k, wcss)]
    """
    results = []
    for k in k_range:
        km = KMeans(
            k=k, seed=seed,
            featuresCol="pca_features",
            predictionCol="cluster_id_tmp",
            maxIter=50, tol=1e-4,
        )
        model = km.fit(stock_pca)
        wcss = float(model.summary.trainingCost)
        results.append((k, wcss))
        log.info("ELBOW | k=%d -> WCSS=%.2f", k, wcss)
    return results


def plot_elbow(elbow_results: List[Tuple[int, float]], output_path: str) -> None:
    """Ve elbow curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r[0] for r in elbow_results]
    wcss = [r[1] for r in elbow_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, wcss, marker="o", linewidth=2, color="#e41a1c", markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method - Optimal k Selection")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT ELBOW | Saved -> %s", output_path)


# ---------------------------------------------------------------------------
# STAGE 3: K-MEANS FINAL
# ---------------------------------------------------------------------------

def kmeans_fit(
    stock_pca: DataFrame,
    k: int,
    seed: int = 42,
) -> Tuple[KMeansModel, DataFrame, float]:
    """Fit K-means tren PC space va tinh Silhouette score.

    Returns:
        model        : KMeansModel
        predictions  : DataFrame voi cot cluster_id
        silhouette   : float
    """
    log.info("STAGE 3 KMEANS  | Fitting k=%d on PC space", k)
    km = KMeans(
        k=k, seed=seed,
        featuresCol="pca_features",
        predictionCol="cluster_id",
        maxIter=100, tol=1e-4,
    )
    model = km.fit(stock_pca)
    preds = model.transform(stock_pca)

    evaluator = ClusteringEvaluator(
        featuresCol="pca_features",
        predictionCol="cluster_id",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    sil = float(evaluator.evaluate(preds))
    log.info("STAGE 3 KMEANS  | Silhouette (AFTER PCA): %.4f", sil)
    return model, preds, sil


# ---------------------------------------------------------------------------
# STAGE 4: CURE CLUSTERING
# ---------------------------------------------------------------------------

def cure_clustering(
    data: np.ndarray,
    k: int = 6,
    n_rep: int = 10,
    alpha: float = 0.2,
    sample_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """CURE algorithm - simplified cho n stocks nho.

    Steps:
      1. Sample random subset
      2. Hierarchical clustering tren sample
      3. Moi cluster -> chon n_rep representatives (greedy farthest-point)
      4. Shrink representatives ve centroid voi factor alpha
      5. Gan moi point con lai vao cluster cua representative gan nhat

    Args:
        data         : np.array shape (n, d) -- PC scores
        k            : so cluster
        n_rep        : so representative per cluster
        alpha        : shrink factor (0=keo het ve tam, 1=khong shrink)
        sample_ratio : ti le sample tu data goc
        seed         : random seed

    Returns:
        labels          : np.array(n,) cluster ID
        representatives : list[k] arrays of representatives
        centroids       : np.array(k, d)
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import cdist

    rng = np.random.RandomState(seed)
    n = len(data)
    sample_size = max(min(n, k * 5), int(n * sample_ratio))
    sample_size = min(sample_size, n)

    log.info("STAGE 4 CURE    | n=%d, k=%d, n_rep=%d, alpha=%.2f, sample=%d",
             n, k, n_rep, alpha, sample_size)

    # Step 1: Sample
    sample_idx = rng.choice(n, sample_size, replace=False)
    sample = data[sample_idx]

    # Step 2: Hierarchical clustering tren sample
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    sample_labels = hc.fit_predict(sample)

    # Step 3: Chon n_rep representatives + shrink
    representatives: List[np.ndarray] = []
    centroids = []
    for cid in range(k):
        mask = sample_labels == cid
        cluster_pts = sample[mask]

        if len(cluster_pts) == 0:
            log.warning("CURE | Cluster %d empty in sample, using sample mean", cid)
            centroid = sample.mean(axis=0)
            reps = sample.mean(axis=0, keepdims=True)
        else:
            centroid = cluster_pts.mean(axis=0)
            if len(cluster_pts) <= n_rep:
                reps = cluster_pts.copy()
            else:
                # Greedy farthest-point: bat dau tu diem xa centroid nhat
                dists_to_centroid = np.linalg.norm(cluster_pts - centroid, axis=1)
                first_idx = int(np.argmax(dists_to_centroid))
                chosen = [first_idx]
                for _ in range(n_rep - 1):
                    chosen_pts = cluster_pts[chosen]
                    min_dists = cdist(cluster_pts, chosen_pts).min(axis=1)
                    next_idx = int(np.argmax(min_dists))
                    chosen.append(next_idx)
                reps = cluster_pts[chosen]

        # Shrink: rep_final = centroid + alpha * (rep - centroid)
        reps_shrunk = centroid + alpha * (reps - centroid)
        representatives.append(reps_shrunk)
        centroids.append(centroid)

    # Step 4: Gan moi point vao cluster cua rep gan nhat
    all_reps = np.vstack(representatives)
    rep_labels = np.repeat(np.arange(k), [len(r) for r in representatives])

    dists = cdist(data, all_reps)
    nearest = dists.argmin(axis=1)
    labels = rep_labels[nearest]

    log.info("STAGE 4 CURE    | Cluster sizes: %s",
             np.bincount(labels, minlength=k).tolist())
    return labels, representatives, np.array(centroids)


def cure_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Tinh silhouette score cho CURE labels (sklearn)."""
    from sklearn.metrics import silhouette_score
    if len(set(labels)) < 2:
        return float("nan")
    sil = float(silhouette_score(data, labels, metric="euclidean"))
    log.info("STAGE 4 CURE    | Silhouette: %.4f", sil)
    return sil


# ---------------------------------------------------------------------------
# STAGE 5: COMPARE METHODS
# ---------------------------------------------------------------------------

def compare_methods(
    symbols: List[str],
    kmeans_labels: np.ndarray,
    cure_labels: np.ndarray,
) -> pd.DataFrame:
    """So sanh K-means vs CURE assignments.

    Returns:
        DataFrame: stock_symbol, kmeans_cluster, cure_cluster, agree
    """
    df = pd.DataFrame({
        "stock_symbol": symbols,
        "kmeans_cluster": kmeans_labels,
        "cure_cluster": cure_labels,
    })

    # Agreement check: 2 thuat toan co the danh nhan cluster khac so
    # nhung "phan vung" giong nhau. Tinh adjusted_rand_score.
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(kmeans_labels, cure_labels)
    nmi = normalized_mutual_info_score(kmeans_labels, cure_labels)

    log.info("COMPARE | Adjusted Rand Index: %.4f", ari)
    log.info("COMPARE | Normalized Mutual Info: %.4f", nmi)
    log.info("COMPARE | (1.0 = identical, 0.0 = random)")

    return df, ari, nmi


# ---------------------------------------------------------------------------
# STAGE 6: VISUALIZATIONS
# ---------------------------------------------------------------------------

def plot_clusters_2d(
    pc_df: pd.DataFrame,
    ev_array: np.ndarray,
    output_path: str,
    cluster_col: str = "cluster_id",
    label_col: str = "stock_symbol",
    title: str = "Stock Clusters in PCA Space (PC1 vs PC2)",
) -> None:
    """Scatter plot 2D voi PC1 vs PC2, mau theo cluster, label = stock_symbol.

    Args:
        pc_df       : pandas DF voi cot pc1, pc2, cluster_col, label_col
        ev_array    : explained variance array (de label axis)
        output_path : duong dan luu PNG
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
               "#ff7f00", "#a65628", "#f781bf", "#999999"]

    n_clusters = pc_df[cluster_col].nunique()
    fig, ax = plt.subplots(figsize=(13, 9))
    for cid in sorted(pc_df[cluster_col].unique()):
        sub = pc_df[pc_df[cluster_col] == cid]
        ax.scatter(
            sub["pc1"], sub["pc2"],
            c=palette[int(cid) % len(palette)],
            s=110, alpha=0.85, edgecolor="black", linewidth=0.5,
            label=f"Cluster {int(cid)} (n={len(sub)})",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row[label_col], (row["pc1"], row["pc2"]),
                fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points",
            )

    ev1 = ev_array[0] * 100 if len(ev_array) > 0 else 0
    ev2 = ev_array[1] * 100 if len(ev_array) > 1 else 0
    ax.set_xlabel(f"PC1 ({ev1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev2:.1f}% variance)")
    ax.set_title(f"{title} | k={n_clusters}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT CLUSTERS | Saved -> %s", output_path)


def plot_cluster_profiles(
    cluster_df: pd.DataFrame,
    output_path: str,
    feature_cols: List[str] = None,
) -> None:
    """Heatmap: mean cua moi indicator goc theo cluster.

    Args:
        cluster_df    : pandas DF voi cot cluster_id + cac feature cols (mean values)
        output_path   : duong dan luu PNG
        feature_cols  : cac feature de plot (default: 7 indicators chinh)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if feature_cols is None:
        feature_cols = [
            "RSI14", "ATR", "ADX20", "bb_width",
            "dist_sma20", "log_return", "volatility_atr_pct",
        ]
    feature_cols = [c for c in feature_cols if c in cluster_df.columns]

    if not feature_cols:
        log.warning("PLOT PROFILES | No feature columns available, skipping")
        return

    pivot = cluster_df[["cluster_id"] + feature_cols].groupby("cluster_id").mean()

    # Z-score normalize moi feature de heatmap so sanh duoc
    pivot_z = (pivot - pivot.mean()) / pivot.std().replace(0, 1)
    pivot_z = pivot_z.fillna(0)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot_z.T.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels([f"Cluster {int(c)}" for c in pivot.index])
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols)

    # Annotate raw values
    for i, feat in enumerate(feature_cols):
        for j, cid in enumerate(pivot.index):
            raw_val = pivot.iloc[j, i]
            ax.text(j, i, f"{raw_val:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    plt.colorbar(im, ax=ax, label="z-score (across clusters)")
    ax.set_title("Cluster Profile Heatmap (raw values, color=z-score)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PLOT PROFILES | Saved -> %s", output_path)
