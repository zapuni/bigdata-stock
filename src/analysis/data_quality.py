import os
import sys
import json
import time
import logging
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, when, isnan, isnull, mean, stddev, 
    min as spark_min, max as spark_max, sum as spark_sum
)

from config.settings import (
    PARQUET_PATH, REPORTS_DIR, SPARK_CONFIG, 
    FEATURE_COLUMNS, QUALITY_THRESHOLDS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_spark_session(app_name="DataQualityAnalysis"):
    """Create and configure Spark session."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(SPARK_CONFIG["master"]) \
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
        .config("spark.driver.maxResultSize", SPARK_CONFIG["max_result_size"]) \
        .config("spark.sql.adaptive.enabled", SPARK_CONFIG["adaptive_enabled"]) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def analyze_missing_values(df, columns, total_rows):
    """
    Analyze missing values for specified columns.
    
    Args:
        df: Spark DataFrame
        columns: List of column names to analyze
        total_rows: Total row count
        
    Returns:
        List of dicts with missing value statistics
    """
    results = []
    
    for col_name in columns:
        if col_name not in df.columns:
            continue
            
        missing_count = df.filter(
            col(col_name).isNull() | isnan(col(col_name))
        ).count()
        
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        
        results.append({
            "column": col_name,
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 4),
            "non_null_count": total_rows - missing_count
        })
    
    return sorted(results, key=lambda x: x["missing_percentage"], reverse=True)


def compute_statistics(df, columns):
    """
    Compute descriptive statistics for numeric columns.
    
    Args:
        df: Spark DataFrame
        columns: List of column names
        
    Returns:
        List of dicts with statistics
    """
    results = []
    
    for col_name in columns:
        if col_name not in df.columns:
            continue
            
        stats = df.select(
            mean(col(col_name)).alias("mean"),
            stddev(col(col_name)).alias("std"),
            spark_min(col(col_name)).alias("min"),
            spark_max(col(col_name)).alias("max")
        ).collect()[0]
        
        results.append({
            "column": col_name,
            "mean": float(stats["mean"]) if stats["mean"] is not None else None,
            "std": float(stats["std"]) if stats["std"] is not None else None,
            "min": float(stats["min"]) if stats["min"] is not None else None,
            "max": float(stats["max"]) if stats["max"] is not None else None
        })
    
    return results


def detect_outliers(df, columns, iqr_multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: Spark DataFrame
        columns: List of column names
        iqr_multiplier: IQR multiplier for bounds (default 1.5)
        
    Returns:
        List of dicts with outlier statistics
    """
    results = []
    total_rows = df.count()
    
    for col_name in columns:
        if col_name not in df.columns:
            continue
            
        quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        
        if len(quantiles) != 2 or None in quantiles:
            continue
            
        q1, q3 = quantiles
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)
        
        outlier_count = df.filter(
            (col(col_name) < lower_bound) | (col(col_name) > upper_bound)
        ).count()
        
        outlier_pct = (outlier_count / total_rows) * 100 if total_rows > 0 else 0
        
        results.append({
            "column": col_name,
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_pct, 4)
        })
    
    return sorted(results, key=lambda x: x["outlier_percentage"], reverse=True)


def assess_completeness(df, feature_columns, threshold=10):
    """
    Assess data completeness by row.
    
    Args:
        df: Spark DataFrame
        feature_columns: List of feature column names
        threshold: Max missing features per row
        
    Returns:
        Dict with completeness metrics
    """
    total_rows = df.count()
    
    valid_columns = [c for c in feature_columns if c in df.columns]
    
    missing_expr = spark_sum(
        sum([when(col(c).isNull() | isnan(col(c)), 1).otherwise(0) for c in valid_columns])
    ).alias("total_missing")
    
    # Count rows with more than threshold missing features
    missing_count_col = sum([
        when(col(c).isNull() | isnan(col(c)), 1).otherwise(0) 
        for c in valid_columns
    ])
    
    incomplete_rows = df.filter(missing_count_col > threshold).count()
    complete_rows = total_rows - incomplete_rows
    
    return {
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "incomplete_rows": incomplete_rows,
        "completeness_score": round((complete_rows / total_rows) * 100, 2) if total_rows > 0 else 0,
        "threshold_missing_features": threshold
    }


def generate_quality_report(missing_stats, statistics, outliers, completeness):
    """
    Generate comprehensive quality report.
    
    Args:
        missing_stats: Missing values analysis results
        statistics: Statistical summary
        outliers: Outlier detection results
        completeness: Completeness assessment
        
    Returns:
        Dict with full report
    """
    high_missing = [m for m in missing_stats if m["missing_percentage"] > QUALITY_THRESHOLDS["max_missing_pct"]]
    high_outliers = [o for o in outliers if o["outlier_percentage"] > 5.0]
    
    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_rows": completeness["total_rows"],
            "total_features_analyzed": len(missing_stats),
            "completeness_score": completeness["completeness_score"],
            "features_high_missing": len(high_missing),
            "features_high_outliers": len(high_outliers)
        },
        "missing_values": missing_stats,
        "statistics": statistics,
        "outliers": outliers,
        "completeness": completeness,
        "recommendations": generate_recommendations(missing_stats, outliers, completeness)
    }


def generate_recommendations(missing_stats, outliers, completeness):
    """Generate data quality recommendations."""
    recommendations = []
    
    high_missing = [m for m in missing_stats if m["missing_percentage"] > 20]
    if high_missing:
        recommendations.append({
            "type": "MISSING_DATA",
            "severity": "HIGH",
            "message": f"{len(high_missing)} features have >20% missing values",
            "affected_columns": [m["column"] for m in high_missing[:5]]
        })
    
    high_outliers = [o for o in outliers if o["outlier_percentage"] > 10]
    if high_outliers:
        recommendations.append({
            "type": "OUTLIERS",
            "severity": "MEDIUM",
            "message": f"{len(high_outliers)} features have >10% outliers",
            "affected_columns": [o["column"] for o in high_outliers[:5]]
        })
    
    if completeness["completeness_score"] < QUALITY_THRESHOLDS["min_completeness_score"]:
        recommendations.append({
            "type": "COMPLETENESS",
            "severity": "HIGH",
            "message": f"Completeness score {completeness['completeness_score']}% is below threshold"
        })
    
    if not recommendations:
        recommendations.append({
            "type": "OK",
            "severity": "INFO",
            "message": "Data quality is within acceptable thresholds"
        })
    
    return recommendations


def save_report(report, output_dir=None):
    """Save quality report to files."""
    output_dir = output_dir or REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    json_path = os.path.join(output_dir, "data_quality_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report: {json_path}")
    
    # Save text summary
    txt_path = os.path.join(output_dir, "data_quality_summary.txt")
    with open(txt_path, "w") as f:
        f.write("DATA QUALITY ASSESSMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {report['generated_at']}\n\n")
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        for key, value in report["summary"].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("TOP 10 COLUMNS WITH MISSING VALUES\n")
        f.write("-" * 40 + "\n")
        for item in report["missing_values"][:10]:
            f.write(f"  {item['column']}: {item['missing_percentage']:.2f}%\n")
        f.write("\n")
        f.write("TOP 10 COLUMNS WITH OUTLIERS\n")
        f.write("-" * 40 + "\n")
        for item in report["outliers"][:10]:
            f.write(f"  {item['column']}: {item['outlier_percentage']:.2f}%\n")
        f.write("\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for rec in report["recommendations"]:
            f.write(f"  [{rec['severity']}] {rec['message']}\n")
    
    logger.info(f"Saved text summary: {txt_path}")
    
    return json_path, txt_path


def run_analysis(input_path=None, output_dir=None):
    """
    Run full data quality analysis pipeline.
    
    Args:
        input_path: Path to Parquet data
        output_dir: Directory for reports
        
    Returns:
        Quality report dict
    """
    input_path = input_path or PARQUET_PATH
    output_dir = output_dir or REPORTS_DIR
    
    if not os.path.exists(input_path):
        logger.error(f"Data path not found: {input_path}")
        return None
    
    spark = get_spark_session()
    
    try:
        start_time = time.time()
        
        logger.info(f"Loading data from {input_path}...")
        df = spark.read.parquet(input_path)
        df.cache()
        
        total_rows = df.count()
        logger.info(f"Loaded {total_rows:,} rows")
        
        # Get available feature columns
        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        logger.info(f"Analyzing {len(available_features)} features...")
        
        # Run analyses
        logger.info("Analyzing missing values...")
        missing_stats = analyze_missing_values(df, available_features, total_rows)
        
        logger.info("Computing statistics...")
        statistics = compute_statistics(df, available_features[:20])
        
        logger.info("Detecting outliers...")
        key_features = ["close", "RSI14", "macd1226", "ATR", "BETA", "volume"]
        outlier_features = [f for f in key_features if f in df.columns]
        outliers = detect_outliers(df, outlier_features)
        
        logger.info("Assessing completeness...")
        completeness = assess_completeness(df, available_features)
        
        # Generate report
        report = generate_quality_report(missing_stats, statistics, outliers, completeness)
        
        # Save reports
        save_report(report, output_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        df.unpersist()
        return report
        
    finally:
        spark.stop()


if __name__ == "__main__":
    report = run_analysis()
    if report:
        logger.info("Data quality analysis completed successfully")
        logger.info(f"Completeness score: {report['summary']['completeness_score']}%")
    else:
        logger.error("Analysis failed")
        sys.exit(1)
