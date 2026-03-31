"""
Runner for PCA + Clustering Pipeline -- Phase 2

Pipeline:
    1. Compute stock profiles    : 63.9M rows -> 101 mean vectors
    2. PCA full fit              : Tinh explained variance, find k_optimal
    3. PCA final fit             : Pipeline (assembler + scaler + PCA(k_optimal))
    4. Loadings analysis         : PC1=Trend, PC2=Volatility, PC3=Momentum
    5. K-means baseline (raw)    : Silhouette TRUOC PCA
    6. K-means elbow             : Tim k toi uu trong PC space
    7. K-means final (k=6)       : Silhouette SAU PCA
    8. CURE clustering           : K-means alternative cho outlier
    9. Compare K-means vs CURE   : Adjusted Rand / NMI
    10. Visualizations           : scree, elbow, scatter, profiles

Usage:
    # Build full pipeline
    python src/run_pca_clustering.py

    # Tuy chinh k cho K-means
    python src/run_pca_clustering.py --kmeans-k 5

    # Bo CURE (chi K-means)
    python src/run_pca_clustering.py --skip-cure

    # Plot lai tu output da co (khong refit)
    python src/run_pca_clustering.py --plot-only
"""

import os
import sys
import time
import argparse
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = SRC_DIR + (os.pathsep + _existing if _existing else "")

import findspark
findspark.init()

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from config.settings import (
    FINAL_PATH, PCA_OUTPUT_PATH, PCA_REPORTS_DIR,
    SPARK_CONFIG, PCA_CONFIG, FEATURE_GROUPS, LOGS_DIR,
)
from analysis.pca_analysis import (
    compute_stock_profiles, fit_pca_full, find_k_optimal,
    fit_pca_pipeline, extract_loadings, interpret_top_loadings,
    plot_scree, explode_pc_columns,
)
from analysis.clustering import (
    silhouette_baseline_raw, kmeans_elbow, plot_elbow,
    kmeans_fit, cure_clustering, cure_silhouette,
    compare_methods, plot_clusters_2d, plot_cluster_profiles,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PCA_OUTPUT_PATH, exist_ok=True)
os.makedirs(PCA_REPORTS_DIR, exist_ok=True)

log = logging.getLogger("stock_pca")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(os.path.join(LOGS_DIR, "pca_clustering.log"))
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_fh)
log.addHandler(_sh)


def _create_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("StockPCA_Clustering")
        .master(SPARK_CONFIG["master"])
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
        .config("spark.driver.maxResultSize", SPARK_CONFIG["max_result_size"])
        .config("spark.sql.adaptive.enabled", SPARK_CONFIG["adaptive_enabled"])
        .config("spark.sql.adaptive.coalescePartitions.enabled",
                SPARK_CONFIG["coalesce_partitions_enabled"])
        .config("spark.memory.fraction", SPARK_CONFIG["spark.memory.fraction"])
        .config("spark.memory.storageFraction", SPARK_CONFIG["spark.memory.storageFraction"])
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG["spark.sql.shuffle.partitions"])
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _save_loadings(loadings_df: pd.DataFrame) -> None:
    """Save loadings CSV + interpretation text."""
    csv_path = os.path.join(PCA_REPORTS_DIR, "pca_loadings.csv")
    loadings_df.to_csv(csv_path)
    log.info("LOADINGS | Saved CSV -> %s", csv_path)

    txt_path = os.path.join(PCA_REPORTS_DIR, "pca_loadings_top.txt")
    interp = interpret_top_loadings(loadings_df, top_n=5, n_pcs=5)
    header = (
        "PCA LOADINGS - TOP 5 FEATURES PER PC\n"
        "=" * 60 + "\n"
        "Loading lon (positive/negative) = feature dong gop nhieu cho PC.\n"
        "PC1 thuong = 'Trend Factor'   (sma/ema/HT_TRENDLINE).\n"
        "PC2 thuong = 'Volatility Factor' (ATR/Trange/bb_width).\n"
        "PC3 thuong = 'Momentum Factor'  (RSI/WILLR/STOCH).\n"
    )
    with open(txt_path, "w") as f:
        f.write(header)
        f.write(interp)
        f.write("\n")
    log.info("LOADINGS | Saved interpretation -> %s", txt_path)


def _save_silhouette_summary(
    sil_raw: float, sil_pca: float, sil_cure: float,
    k_optimal: int, k_kmeans: int, ev_at_k: float,
    ari: float, nmi: float,
) -> None:
    txt_path = os.path.join(PCA_REPORTS_DIR, "silhouette_comparison.txt")
    with open(txt_path, "w") as f:
        f.write("SILHOUETTE SCORE COMPARISON\n")
        f.write("=" * 50 + "\n")
        f.write(f"BEFORE PCA (raw {len(FEATURE_GROUPS['pca_input'])} features):"
                f" {sil_raw:.4f}\n")
        f.write(f"AFTER PCA  ({k_optimal} PCs, K-means k={k_kmeans}):"
                f" {sil_pca:.4f}\n")
        f.write(f"AFTER PCA  ({k_optimal} PCs, CURE     k={k_kmeans}):"
                f" {sil_cure:.4f}\n")
        improvement = (sil_pca - sil_raw) / max(abs(sil_raw), 1e-9) * 100
        f.write(f"\nImprovement (PCA vs raw): {improvement:+.1f}%\n")
        f.write(f"k_optimal (>= 90% variance): {k_optimal}\n")
        f.write(f"Explained Variance at k={k_optimal}: {ev_at_k*100:.2f}%\n")
        f.write("\nK-means vs CURE agreement:\n")
        f.write(f"  Adjusted Rand Index    : {ari:.4f}  (1.0 = identical)\n")
        f.write(f"  Normalized Mutual Info : {nmi:.4f}\n")
    log.info("SUMMARY | Saved silhouette comparison -> %s", txt_path)


def main():
    parser = argparse.ArgumentParser(description="PCA + Clustering Pipeline")
    parser.add_argument("--kmeans-k", type=int, default=PCA_CONFIG["kmeans_k"],
                        help=f"K cho K-means (default: {PCA_CONFIG['kmeans_k']})")
    parser.add_argument("--variance-threshold", type=float,
                        default=PCA_CONFIG["variance_threshold"],
                        help="Cumulative variance threshold de chon k_optimal")
    parser.add_argument("--skip-cure", action="store_true",
                        help="Bo CURE clustering (chi K-means)")
    parser.add_argument("--skip-elbow", action="store_true",
                        help="Bo elbow curve (tiet kiem thoi gian)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("STOCK BIG DATA - PCA + CLUSTERING PIPELINE")
    log.info("=" * 70)

    spark = _create_spark()
    log.info("Spark: %s | cores: %d",
             spark.version, spark.sparkContext.defaultParallelism)
    log.info("Input    : %s", FINAL_PATH)
    log.info("Output   : %s", PCA_OUTPUT_PATH)
    log.info("Reports  : %s", PCA_REPORTS_DIR)

    pca_cols = FEATURE_GROUPS["pca_input"]
    seed = PCA_CONFIG["random_seed"]

    t_total = time.time()

    try:
        # ----- STAGE 1: Compute stock profiles -----
        profiles = compute_stock_profiles(spark, FINAL_PATH, pca_cols)
        profiles_path = os.path.join(PCA_OUTPUT_PATH, "stock_profiles")
        profiles.write.mode("overwrite").parquet(profiles_path)
        log.info("STAGE 1 PROFILES | Saved -> %s", profiles_path)

        # ----- STAGE 2-3: PCA full fit + find k_optimal -----
        prep_full_model, ev_array, cumsum = fit_pca_full(profiles, pca_cols)
        k_optimal = find_k_optimal(cumsum, threshold=args.variance_threshold)
        ev_at_k = float(cumsum[k_optimal - 1])

        # ----- STAGE 4: PCA final fit (k=k_optimal) -----
        pca_model, stock_pca = fit_pca_pipeline(profiles, pca_cols, k=k_optimal)
        model_path = os.path.join(PCA_OUTPUT_PATH, "pca_model")
        pca_model.write().overwrite().save(model_path)
        log.info("STAGE 4 MODEL | Saved -> %s", model_path)

        # Cache cho clustering reuse
        stock_pca = explode_pc_columns(stock_pca, k=k_optimal)
        stock_pca.cache()

        # ----- STAGE 5: Loadings -----
        loadings_df = extract_loadings(pca_model, pca_cols)
        _save_loadings(loadings_df)

        # ----- STAGE 6: Silhouette baseline (TRUOC PCA) -----
        # Reuse assembler + scaler tu pca_model
        scaled_df = pca_model.stages[1].transform(
            pca_model.stages[0].transform(profiles)
        )
        sil_raw, _ = silhouette_baseline_raw(scaled_df, k=args.kmeans_k, seed=seed)

        # ----- STAGE 7: Elbow method -----
        if not args.skip_elbow:
            k_lo, k_hi = PCA_CONFIG["elbow_k_range"]
            elbow_results = kmeans_elbow(stock_pca, range(k_lo, k_hi), seed=seed)
            elbow_path = os.path.join(PCA_REPORTS_DIR, "elbow_curve.png")
            plot_elbow(elbow_results, elbow_path)

            elbow_csv = os.path.join(PCA_REPORTS_DIR, "elbow_wcss.csv")
            pd.DataFrame(elbow_results, columns=["k", "wcss"]).to_csv(
                elbow_csv, index=False
            )

        # ----- STAGE 8: K-means final -----
        kmeans_model, kmeans_preds, sil_pca = kmeans_fit(
            stock_pca, k=args.kmeans_k, seed=seed
        )

        # Collect 101 stocks ve driver de save + CURE
        pc_cols = [f"pc{i+1}" for i in range(k_optimal)]
        cluster_pdf = (
            kmeans_preds
            .select("stock_symbol", "cluster_id", "trading_days", *pc_cols)
            .toPandas()
            .sort_values("stock_symbol")
            .reset_index(drop=True)
        )

        # ----- STAGE 9: CURE -----
        if not args.skip_cure:
            data_matrix = cluster_pdf[pc_cols].values
            symbols = cluster_pdf["stock_symbol"].tolist()
            cure_labels, _, _ = cure_clustering(
                data_matrix,
                k=args.kmeans_k,
                n_rep=PCA_CONFIG["cure_n_rep"],
                alpha=PCA_CONFIG["cure_alpha"],
                seed=seed,
            )
            sil_cure = cure_silhouette(data_matrix, cure_labels)
            cluster_pdf["cure_cluster"] = cure_labels
            kmeans_labels = cluster_pdf["cluster_id"].values
            _, ari, nmi = compare_methods(symbols, kmeans_labels, cure_labels)
        else:
            sil_cure, ari, nmi = float("nan"), float("nan"), float("nan")

        # ----- STAGE 10: Save outputs -----
        cluster_csv = os.path.join(PCA_REPORTS_DIR, "cluster_assignments.csv")
        cluster_pdf.to_csv(cluster_csv, index=False)
        log.info("OUTPUT | Cluster assignments -> %s", cluster_csv)

        # Save Spark Parquet (cho dashboard/downstream)
        cluster_parquet = os.path.join(PCA_OUTPUT_PATH, "cluster_labels")
        spark.createDataFrame(cluster_pdf).write.mode("overwrite").parquet(cluster_parquet)
        log.info("OUTPUT | Cluster labels Parquet -> %s", cluster_parquet)

        # ----- STAGE 11: Visualizations -----
        scree_path = os.path.join(PCA_REPORTS_DIR, "scree_plot.png")
        plot_scree(ev_array, cumsum, k_optimal, scree_path,
                   threshold=args.variance_threshold)

        scatter_path = os.path.join(PCA_REPORTS_DIR, "pca_scatter_clusters.png")
        plot_clusters_2d(cluster_pdf, ev_array, scatter_path,
                         cluster_col="cluster_id")

        # Cluster profile heatmap: doc lai mean features per cluster
        # Join cluster labels voi stock profiles de get raw indicator means
        profiles_pdf = profiles.toPandas()
        merged = profiles_pdf.merge(
            cluster_pdf[["stock_symbol", "cluster_id"]],
            on="stock_symbol", how="left",
        )
        profiles_path_png = os.path.join(PCA_REPORTS_DIR, "cluster_profiles.png")
        plot_cluster_profiles(merged, profiles_path_png)

        # ----- STAGE 12: Summary -----
        _save_silhouette_summary(
            sil_raw, sil_pca, sil_cure,
            k_optimal, args.kmeans_k, ev_at_k, ari, nmi,
        )

        elapsed = time.time() - t_total
        log.info("=" * 70)
        log.info("PIPELINE COMPLETED in %.1f min (%.0f sec)",
                 elapsed / 60, elapsed)
        log.info("=" * 70)
        log.info("Silhouette BEFORE PCA : %.4f", sil_raw)
        log.info("Silhouette AFTER PCA  : %.4f (K-means)", sil_pca)
        if not args.skip_cure:
            log.info("Silhouette AFTER PCA  : %.4f (CURE)", sil_cure)
            log.info("K-means vs CURE ARI   : %.4f", ari)
        log.info("k_optimal (>=90%% var) : %d (giu %.2f%%)",
                 k_optimal, ev_at_k * 100)
        log.info("=" * 70)

    except Exception:
        log.exception("PIPELINE FAILED")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
