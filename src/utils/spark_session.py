import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import findspark
findspark.init()

from pyspark.sql import SparkSession
from config.settings import SPARK_CONFIG


class SparkSessionFactory:
    """Factory class for creating and managing Spark sessions."""
    
    _instance = None
    
    @classmethod
    def get_session(cls, app_name=None):
        """
        Get or create a Spark session.
        
        Args:
            app_name: Optional application name override
            
        Returns:
            SparkSession instance
        """
        if cls._instance is None or cls._instance._jsc is None:
            cls._instance = cls._create_session(app_name)
        return cls._instance
    
    @classmethod
    def _create_session(cls, app_name=None):
        """Create a new Spark session with project configuration."""
        name = app_name or SPARK_CONFIG["app_name"]
        
        spark = SparkSession.builder \
            .appName(name) \
            .master(SPARK_CONFIG["master"]) \
            .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
            .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
            .config("spark.driver.maxResultSize", SPARK_CONFIG["max_result_size"]) \
            .config("spark.sql.adaptive.enabled", SPARK_CONFIG["adaptive_enabled"]) \
            .config("spark.sql.adaptive.coalescePartitions.enabled", 
                    SPARK_CONFIG["coalesce_partitions_enabled"]) \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
    
    @classmethod
    def stop_session(cls):
        """Stop the current Spark session."""
        if cls._instance is not None:
            cls._instance.stop()
            cls._instance = None


def get_spark_session(app_name=None):
    """
    Convenience function to get a Spark session.
    
    Args:
        app_name: Optional application name
        
    Returns:
        SparkSession instance
    """
    return SparkSessionFactory.get_session(app_name)


def stop_spark_session():
    """Convenience function to stop the Spark session."""
    SparkSessionFactory.stop_session()
