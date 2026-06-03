from __future__ import annotations

import contextlib
import json
from datetime import datetime

import pymysql

from src.config import Settings


RUN_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_run_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    started_at TIMESTAMP NULL,
    finished_at TIMESTAMP NULL,
    status VARCHAR(32) NOT NULL,
    use_s3 BOOLEAN NOT NULL,
    kafka_enabled BOOLEAN NOT NULL,
    max_rows_per_dataset BIGINT NULL,
    producer_batch_size INT NULL,
    producer_batch_delay_seconds DOUBLE NULL,
    consumer_timeout_seconds INT NULL,
    datasets_count INT NULL,
    total_row_count BIGINT NULL,
    total_size_bytes BIGINT NULL,
    error_message TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY run_id (run_id)
);
"""

DATASET_DDL = """
CREATE TABLE IF NOT EXISTS dataset_run_summary (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(128) NOT NULL,
    layer_name VARCHAR(64) NOT NULL,
    storage_path TEXT NOT NULL,
    s3_uri TEXT NULL,
    row_count BIGINT NULL,
    file_count INT NULL,
    size_bytes BIGINT NULL,
    column_count INT NULL,
    schema_hash VARCHAR(64) NULL,
    sample_files_json TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY run_dataset_layer (run_id, dataset_name, layer_name)
);
"""

COLUMN_DDL = """
CREATE TABLE IF NOT EXISTS column_stats (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(128) NOT NULL,
    layer_name VARCHAR(64) NOT NULL,
    column_name VARCHAR(256) NOT NULL,
    data_type VARCHAR(128) NULL,
    null_count BIGINT NULL,
    null_ratio DOUBLE NULL,
    min_value TEXT NULL,
    max_value TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _connect(settings: Settings, database: str | None = None):
    return pymysql.connect(
        host=settings.rds_host,
        port=settings.rds_port,
        user=settings.rds_user,
        password=settings.rds_password,
        database=database,
        charset="utf8mb4",
        autocommit=False,
    )


def _ensure_database(settings: Settings) -> None:
    with contextlib.closing(_connect(settings)) as admin_conn:
        with admin_conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.rds_database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        admin_conn.commit()


def register_pipeline_run(
    settings: Settings,
    run_id: str,
    *,
    status: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    kafka_enabled: bool = True,
    use_s3: bool = True,
    max_rows_per_dataset: int | None = None,
    producer_batch_size: int | None = None,
    producer_batch_delay_seconds: float | None = None,
    consumer_timeout_seconds: int | None = None,
    datasets_count: int | None = None,
    total_row_count: int | None = None,
    total_size_bytes: int | None = None,
    error_message: str | None = None,
) -> None:
    _ensure_database(settings)

    with contextlib.closing(_connect(settings, settings.rds_database)) as conn:
        _ensure_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_run_summary (
                    run_id,
                    started_at,
                    finished_at,
                    status,
                    use_s3,
                    kafka_enabled,
                    max_rows_per_dataset,
                    producer_batch_size,
                    producer_batch_delay_seconds,
                    consumer_timeout_seconds,
                    datasets_count,
                    total_row_count,
                    total_size_bytes,
                    error_message
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    started_at=VALUES(started_at),
                    finished_at=VALUES(finished_at),
                    status=VALUES(status),
                    use_s3=VALUES(use_s3),
                    kafka_enabled=VALUES(kafka_enabled),
                    max_rows_per_dataset=VALUES(max_rows_per_dataset),
                    producer_batch_size=VALUES(producer_batch_size),
                    producer_batch_delay_seconds=VALUES(producer_batch_delay_seconds),
                    consumer_timeout_seconds=VALUES(consumer_timeout_seconds),
                    datasets_count=VALUES(datasets_count),
                    total_row_count=VALUES(total_row_count),
                    total_size_bytes=VALUES(total_size_bytes),
                    error_message=VALUES(error_message)
                """,
                (
                    run_id,
                    started_at,
                    finished_at,
                    status,
                    use_s3,
                    kafka_enabled,
                    max_rows_per_dataset,
                    producer_batch_size,
                    producer_batch_delay_seconds,
                    consumer_timeout_seconds,
                    datasets_count,
                    total_row_count,
                    total_size_bytes,
                    error_message,
                ),
            )
        conn.commit()


def register_dataset_run(
    settings: Settings,
    dataset: str,
    layer: str,
    storage_path: str,
    *,
    run_id: str,
    summary: dict[str, object],
) -> None:
    _ensure_database(settings)
    with contextlib.closing(_connect(settings, settings.rds_database)) as conn:
        _ensure_tables(conn)
        sample_files_json = _json_or_none(summary.get("sample_files"))
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset_run_summary (
                    run_id,
                    dataset_name,
                    layer_name,
                    storage_path,
                    s3_uri,
                    row_count,
                    file_count,
                    size_bytes,
                    column_count,
                    schema_hash,
                    sample_files_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    storage_path=VALUES(storage_path),
                    s3_uri=VALUES(s3_uri),
                    row_count=VALUES(row_count),
                    file_count=VALUES(file_count),
                    size_bytes=VALUES(size_bytes),
                    column_count=VALUES(column_count),
                    schema_hash=VALUES(schema_hash),
                    sample_files_json=VALUES(sample_files_json)
                """,
                (
                    run_id,
                    dataset,
                    layer,
                    storage_path,
                    summary.get("s3_uri"),
                    summary.get("row_count"),
                    summary.get("file_count"),
                    summary.get("size_bytes"),
                    summary.get("column_count"),
                    summary.get("schema_hash"),
                    sample_files_json,
                ),
            )
        conn.commit()


def register_column_stats(
    settings: Settings,
    dataset: str,
    layer: str,
    *,
    run_id: str,
    row_count: int,
    column_stats: list[dict[str, object]],
) -> None:
    if not column_stats:
        return
    _ensure_database(settings)
    with contextlib.closing(_connect(settings, settings.rds_database)) as conn:
        _ensure_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM column_stats WHERE run_id=%s AND dataset_name=%s AND layer_name=%s",
                (run_id, dataset, layer),
            )
            rows = []
            for stat in column_stats:
                null_count = stat.get("null_count")
                null_ratio = None
                if isinstance(null_count, int) and row_count:
                    null_ratio = float(null_count) / float(row_count)
                rows.append(
                    (
                        run_id,
                        dataset,
                        layer,
                        stat.get("name"),
                        stat.get("type"),
                        null_count,
                        null_ratio,
                        stat.get("min"),
                        stat.get("max"),
                    )
                )
            cur.executemany(
                """
                INSERT INTO column_stats (
                    run_id,
                    dataset_name,
                    layer_name,
                    column_name,
                    data_type,
                    null_count,
                    null_ratio,
                    min_value,
                    max_value
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()


def _ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(RUN_DDL)
        cur.execute(DATASET_DDL)
        cur.execute(COLUMN_DDL)
    conn.commit()


def _json_or_none(value: object | None) -> str | None:
    if value in (None, {}, []):
        return None
    return json.dumps(value)
