from __future__ import annotations

import logging
import os

from src.config import PROJECT_ROOT
from pyspark.sql import SparkSession

LOGGER = logging.getLogger(__name__)


def build_spark(app_name: str = "music-recommender-data-pipeline") -> SparkSession:
    hadoop_home = PROJECT_ROOT / "tools" / "hadoop"
    if (hadoop_home / "bin" / "winutils.exe").exists():
        os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
        os.environ.setdefault("hadoop.home.dir", str(hadoop_home))
        os.environ["PATH"] = f"{hadoop_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        LOGGER.info("Configured local Hadoop tools from %s", hadoop_home)
    warehouse_dir = (PROJECT_ROOT / "spark_warehouse").resolve().as_uri()
    LOGGER.info("Creating Spark session %s", app_name)
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .getOrCreate()
    )
    LOGGER.info("Spark session ready with warehouse %s", warehouse_dir)
    return spark
