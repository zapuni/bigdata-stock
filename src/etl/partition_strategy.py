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
from pyspark.sql.functions import col, year, month, countDistinct

from config.settings import (
    PARQUET_PATH, REPORTS_DIR, SPARK_CONFIG,
    FEATURE_COLUMNS, PARTITION_COLUMNS, get_optimized_schema
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_spark_session(app_name="SchemaPartitionStrategy"):
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


def extract_metadata(df):
    """
    Extract metadata from DataFrame.
    
    Args:
        df: Spark DataFrame
        
    Returns:
        Dict with metadata
    """
    total_rows = df.count()
    
    # Get stock symbols
    stocks = df.select("stock_symbol").distinct().collect()
    stock_list = sorted([row.stock_symbol for row in stocks if row.stock_symbol])
    
    # Get year range
    years = df.select("year").distinct().orderBy("year").collect()
    year_list = [row.year for row in years if row.year]
    
    # Get partition statistics
    partition_stats = df.groupBy("year", "month").count().collect()
    
    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "stocks": stock_list,
        "stock_count": len(stock_list),
        "year_range": [min(year_list), max(year_list)] if year_list else [],
        "partition_count": len(partition_stats)
    }


def test_query_performance(spark, parquet_path):
    """
    Test query performance on partitioned data.
    
    Args:
        spark: SparkSession
        parquet_path: Path to Parquet data
        
    Returns:
        List of query performance results
    """
    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("stocks")
    
    queries = [
        {
            "name": "single_stock_single_month",
            "description": "Query single stock for one month",
            "sql": """
                SELECT COUNT(*) as count
                FROM stocks
                WHERE year = 2020 AND month = 1 
                AND stock_symbol LIKE 'ACC%'
            """
        },
        {
            "name": "all_stocks_single_year",
            "description": "Aggregate all stocks for one year",
            "sql": """
                SELECT stock_symbol, COUNT(*) as count
                FROM stocks
                WHERE year = 2020
                GROUP BY stock_symbol
            """
        },
        {
            "name": "time_series_single_stock",
            "description": "Time series for single stock",
            "sql": """
                SELECT timestamp, close, RSI14, macd1226
                FROM stocks
                WHERE stock_symbol LIKE 'HDFC%'
                AND year = 2020
                ORDER BY timestamp
                LIMIT 1000
            """
        }
    ]
    
    results = []
    for query in queries:
        start_time = time.time()
        try:
            result = spark.sql(query["sql"]).collect()
            elapsed = time.time() - start_time
            results.append({
                "name": query["name"],
                "description": query["description"],
                "elapsed_seconds": round(elapsed, 4),
                "status": "SUCCESS",
                "row_count": len(result)
            })
        except Exception as e:
            results.append({
                "name": query["name"],
                "description": query["description"],
                "elapsed_seconds": 0,
                "status": "FAILED",
                "error": str(e)
            })
    
    return results


def generate_schema_documentation(schema, metadata, performance_results):
    """
    Generate comprehensive schema documentation.
    
    Args:
        schema: StructType schema
        metadata: Metadata dict
        performance_results: Query performance results
        
    Returns:
        Documentation dict
    """
    field_groups = {
        "primary_key": ["stock_symbol", "timestamp"],
        "partition_keys": ["year", "month"],
        "ohlcv": ["open", "high", "low", "close", "volume"],
        "moving_averages": ["sma5", "sma10", "sma15", "sma20", "ema5", "ema10", "ema15", "ema20"],
        "bollinger_bands": ["upperband", "middleband", "lowerband"],
        "trend_indicators": ["ht_trendline", "kama10", "kama20", "kama30", "sar", "trima5", "trima10", "trima20"],
        "momentum": ["adx5", "adx10", "adx20", "apo", "cci5", "cci10", "cci15"],
        "macd": ["macd510", "macd520", "macd1020", "macd1520", "macd1226"],
        "rate_of_change": ["mfi", "mom10", "mom15", "mom20", "roc5", "roc10", "roc20", "ppo"],
        "rsi": ["rsi14", "rsi8"],
        "stochastic": ["slowk", "slowd", "fastk", "fastd", "fastksr", "fastdsr"],
        "oscillators": ["ultosc", "willr"],
        "volatility": ["atr", "trange", "typprice"],
        "others": ["ht_dcperiod", "beta"]
    }
    
    return {
        "generated_at": datetime.now().isoformat(),
        "table_name": "stock_market_data",
        "storage_format": "Parquet",
        "compression": "Snappy",
        "metadata": metadata,
        "schema": {
            "total_fields": len(schema.fields),
            "field_groups": field_groups,
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.dataType),
                    "nullable": field.nullable
                }
                for field in schema.fields
            ]
        },
        "partition_strategy": {
            "partition_columns": PARTITION_COLUMNS,
            "directory_structure": "year={year}/month={month}/stock_symbol={symbol}/",
            "benefits": [
                "Time-based queries scan only relevant year/month partitions",
                "Stock-based queries scan only relevant stock partitions",
                "Parallel processing enabled per partition"
            ]
        },
        "query_patterns": [
            {
                "name": "Single Stock Analysis",
                "example": "SELECT * FROM stocks WHERE stock_symbol = 'HDFC' AND year = 2022",
                "partitions_scanned": "12 (1 stock x 12 months)"
            },
            {
                "name": "Cross-Stock Comparison",
                "example": "SELECT stock_symbol, AVG(close) FROM stocks WHERE year = 2022 AND month = 6 GROUP BY stock_symbol",
                "partitions_scanned": "N (N stocks x 1 month)"
            },
            {
                "name": "Time Series Analysis",
                "example": "SELECT date, close, rsi14 FROM stocks WHERE stock_symbol = 'RELIANCE' AND year BETWEEN 2020 AND 2022",
                "partitions_scanned": "36 (1 stock x 36 months)"
            }
        ],
        "performance_tests": performance_results,
        "indexing_recommendations": [
            "(timestamp, stock_symbol) for chronological queries",
            "(stock_symbol, year, month) for stock-specific analysis",
            "(rsi14, macd1226) for signal-based screening"
        ]
    }


def save_documentation(doc, output_dir=None):
    """Save schema documentation to files."""
    output_dir = output_dir or REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "schema_documentation.json")
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=2)
    logger.info(f"Saved JSON documentation: {json_path}")
    
    # Save text documentation
    txt_path = os.path.join(output_dir, "schema_documentation.txt")
    with open(txt_path, "w") as f:
        f.write("STOCK MARKET DATABASE SCHEMA DOCUMENTATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {doc['generated_at']}\n\n")
        
        f.write("1. DATA MODEL\n")
        f.write("-" * 40 + "\n")
        f.write(f"Table: {doc['table_name']}\n")
        f.write(f"Storage: {doc['storage_format']}\n")
        f.write(f"Compression: {doc['compression']}\n")
        f.write(f"Total Rows: {doc['metadata']['total_rows']:,}\n")
        f.write(f"Total Columns: {doc['metadata']['total_columns']}\n")
        f.write(f"Stocks: {doc['metadata']['stock_count']}\n\n")
        
        f.write("2. SCHEMA STRUCTURE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Fields: {doc['schema']['total_fields']}\n\n")
        f.write("Field Groups:\n")
        for group, fields in doc["schema"]["field_groups"].items():
            f.write(f"  {group}: {len(fields)} fields\n")
        f.write("\n")
        
        f.write("3. PARTITIONING STRATEGY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Partition Columns: {', '.join(doc['partition_strategy']['partition_columns'])}\n")
        f.write(f"Directory Structure: {doc['partition_strategy']['directory_structure']}\n\n")
        f.write("Benefits:\n")
        for benefit in doc["partition_strategy"]["benefits"]:
            f.write(f"  - {benefit}\n")
        f.write("\n")
        
        f.write("4. QUERY PATTERNS\n")
        f.write("-" * 40 + "\n")
        for pattern in doc["query_patterns"]:
            f.write(f"\n{pattern['name']}:\n")
            f.write(f"  SQL: {pattern['example']}\n")
            f.write(f"  Partitions: {pattern['partitions_scanned']}\n")
        f.write("\n")
        
        f.write("5. PERFORMANCE TESTS\n")
        f.write("-" * 40 + "\n")
        for test in doc["performance_tests"]:
            status = test["status"]
            time_str = f"{test['elapsed_seconds']:.4f}s" if status == "SUCCESS" else "N/A"
            f.write(f"  {test['name']}: {time_str} ({status})\n")
        f.write("\n")
        
        f.write("6. INDEXING RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        for rec in doc["indexing_recommendations"]:
            f.write(f"  - {rec}\n")
    
    logger.info(f"Saved text documentation: {txt_path}")
    
    return json_path, txt_path


def run_schema_analysis(input_path=None, output_dir=None):
    """
    Run schema analysis and documentation pipeline.
    
    Args:
        input_path: Path to Parquet data
        output_dir: Directory for documentation
        
    Returns:
        Documentation dict
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
        
        logger.info("Extracting metadata...")
        metadata = extract_metadata(df)
        logger.info(f"Found {metadata['stock_count']} stocks, {metadata['total_rows']:,} rows")
        
        logger.info("Testing query performance...")
        performance_results = test_query_performance(spark, input_path)
        
        logger.info("Generating documentation...")
        schema = get_optimized_schema()
        doc = generate_schema_documentation(schema, metadata, performance_results)
        
        save_documentation(doc, output_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Schema analysis completed in {elapsed_time:.2f} seconds")
        
        return doc
        
    finally:
        spark.stop()


if __name__ == "__main__":
    doc = run_schema_analysis()
    if doc:
        logger.info("Schema documentation generated successfully")
    else:
        logger.error("Schema analysis failed")
        sys.exit(1)
