from __future__ import annotations

import contextlib

import psycopg2

from src.config import Settings


DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    layer_name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def register_pipeline_output(settings: Settings, dataset: str, layer: str, storage_path: str) -> None:
    """Persist pipeline metadata in an RDS-compatible PostgreSQL database."""
    with contextlib.closing(psycopg2.connect(settings.rds_dsn)) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute(
                """
                INSERT INTO pipeline_runs (dataset_name, layer_name, storage_path)
                VALUES (%s, %s, %s)
                """,
                (dataset, layer, storage_path),
            )
        conn.commit()
