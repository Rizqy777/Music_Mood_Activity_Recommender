from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from src.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AthenaTableSpec:
    table_name: str
    s3_location: str
    local_path: Path
    description: str


def register_parquet_tables(settings: Settings, specs: Iterable[AthenaTableSpec]) -> list[str]:
    """Register local Parquet outputs as external Glue/Athena tables over S3."""
    if not settings.use_s3:
        LOGGER.warning("Athena/Glue catalog registration skipped because USE_S3=false")
        return []

    glue = boto3.client("glue", region_name=settings.aws_region)
    _ensure_database(glue, settings.athena_database)
    registered: list[str] = []
    for spec in specs:
        schema = _read_parquet_schema(spec.local_path)
        if not schema:
            LOGGER.warning("Skipping Athena table %s because no parquet schema was found at %s", spec.table_name, spec.local_path)
            continue
        table_name = _safe_table_name(spec.table_name)
        table_input = _build_table_input(table_name, spec.s3_location, schema, spec.description)
        try:
            glue.create_table(DatabaseName=settings.athena_database, TableInput=table_input)
            LOGGER.info("Created Glue/Athena table %s.%s", settings.athena_database, table_name)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) != "AlreadyExistsException":
                raise
            glue.update_table(DatabaseName=settings.athena_database, TableInput=table_input)
            LOGGER.info("Updated Glue/Athena table %s.%s", settings.athena_database, table_name)
        registered.append(table_name)
    return registered


def run_validation_queries(settings: Settings, table_names: Iterable[str]) -> dict[str, str]:
    """Run lightweight Athena COUNT queries to verify registered tables are queryable."""
    if not settings.use_s3:
        return {}
    athena = boto3.client("athena", region_name=settings.aws_region)
    results: dict[str, str] = {}
    for table_name in table_names:
        query = f"SELECT COUNT(*) AS rows_count FROM {settings.athena_database}.{table_name}"
        execution_id = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": settings.athena_database},
            ResultConfiguration={"OutputLocation": settings.athena_results_s3},
            WorkGroup=settings.athena_workgroup,
        )["QueryExecutionId"]
        _wait_for_query(athena, execution_id)
        results[table_name] = execution_id
        LOGGER.info("Athena validation query OK for %s: %s", table_name, execution_id)
    return results


def specs_from_outputs(settings: Settings, layer: str, outputs: dict[str, Path]) -> list[AthenaTableSpec]:
    specs: list[AthenaTableSpec] = []
    for dataset, local_path in outputs.items():
        specs.append(
            AthenaTableSpec(
                table_name=f"{layer}_{dataset}",
                s3_location=f"s3://{settings.bucket_name}/{layer}/{dataset}/",
                local_path=local_path,
                description=f"{layer} layer table for {dataset}",
            )
        )
    return specs


def _ensure_database(glue, database_name: str) -> None:
    try:
        glue.get_database(Name=database_name)
        return
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code")) != "EntityNotFoundException":
            raise
    glue.create_database(DatabaseInput={"Name": database_name})
    LOGGER.info("Created Glue database %s", database_name)


def _read_parquet_schema(path: Path) -> pa.Schema | None:
    if path.is_file() and path.suffix == ".parquet":
        return pq.read_schema(path)
    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        return None
    return pq.read_schema(parquet_files[0])


def _build_table_input(table_name: str, s3_location: str, schema: pa.Schema, description: str) -> dict:
    columns = [
        {"Name": _safe_column_name(field.name), "Type": _athena_type(field.type)}
        for field in schema
        if not field.name.startswith("__")
    ]
    return {
        "Name": table_name,
        "Description": description,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "classification": "parquet",
            "EXTERNAL": "TRUE",
        },
        "StorageDescriptor": {
            "Columns": columns,
            "Location": s3_location,
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                "Parameters": {"serialization.format": "1"},
            },
        },
    }


def _athena_type(data_type: pa.DataType) -> str:
    if pa.types.is_boolean(data_type):
        return "boolean"
    if pa.types.is_integer(data_type):
        return "bigint"
    if pa.types.is_floating(data_type):
        return "double"
    if pa.types.is_timestamp(data_type):
        return "timestamp"
    if pa.types.is_date(data_type):
        return "date"
    return "string"


def _safe_table_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value).lower()


def _safe_column_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", value).lower()
    if cleaned and cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def _wait_for_query(athena, execution_id: str, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        execution = athena.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED"}:
            reason = execution["Status"].get("StateChangeReason", "unknown")
            raise RuntimeError(f"Athena query {execution_id} ended as {state}: {reason}")
        time.sleep(2)
    raise TimeoutError(f"Athena query {execution_id} did not finish in {timeout_seconds}s")
