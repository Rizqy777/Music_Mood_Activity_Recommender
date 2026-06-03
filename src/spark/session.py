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
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", driver_memory)
    shuffle_partitions = int(os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "400"))
    max_partition_bytes = os.getenv("SPARK_SQL_FILES_MAX_PARTITION_BYTES", "64m")
    adaptive_enabled = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    serializer = os.getenv("SPARK_SERIALIZER", "org.apache.spark.serializer.KryoSerializer")

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.files.maxPartitionBytes", max_partition_bytes)
        .config("spark.sql.adaptive.enabled", str(adaptive_enabled).lower())
        .config("spark.serializer", serializer)
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .getOrCreate()
    )
    LOGGER.info(
        "Spark configs: driver=%s executor=%s shuffle=%s maxPartitionBytes=%s adaptive=%s",
        driver_memory,
        executor_memory,
        shuffle_partitions,
        max_partition_bytes,
        adaptive_enabled,
    )
    LOGGER.info("Spark session ready with warehouse %s", warehouse_dir)
    return spark
