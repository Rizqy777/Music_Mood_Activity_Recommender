from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings, load_settings

LOGGER = logging.getLogger("pipeline")
LOG_DIR = ROOT / "logs"
STAGE_LOGS = {
    "docker": "00_docker.log",
    "storage": "01_storage_s3.log",
    "kafka": "02_kafka.log",
    "bronze": "03_bronze.log",
    "silver": "04_silver.log",
    "gold": "05_gold.log",
    "metadata": "06_metadata.log",
    "s3": "07_s3.log",
    "athena": "08_athena_glue.log",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the music recommender data pipeline.")
    parser.add_argument("--skip-docker", action="store_true", help="Do not start local Docker services.")
    parser.add_argument("--skip-kafka", action="store_true", help="Skip Kafka ingestion and only run Spark layers.")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip MongoDB/RDS metadata registration.")
    parser.add_argument("--skip-gold", action="store_true", help="Skip Gold transformations.")
    parser.add_argument(
        "--auto-deploy-lambda",
        action="store_true",
        help="Auto-deploy the AWS Lambda if missing (requires --lambda-role-arn or AWS_LAMBDA_ROLE_ARN).",
    )
    parser.add_argument(
        "--lambda-role-arn",
        default=None,
        help="IAM role ARN to deploy the Lambda when --auto-deploy-lambda is enabled.",
    )
    parser.add_argument(
        "--run-gold",
        action="store_true",
        help="Run Gold transformations (by default they are skipped).",
    )
    parser.add_argument(
        "--clean-bronze",
        action="store_true",
        help="Delete existing Bronze files before consuming Kafka events for this run.",
    )
    parser.add_argument(
        "--monitor-terminals",
        action="store_true",
        help="Open extra PowerShell windows that stream Docker and pipeline stage logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    configure_logging()
    pipeline = import_pipeline_dependencies(args)
    run_id = os.getenv("PIPELINE_RUN_ID") or str(uuid.uuid4())
    run_started_at = datetime.now(timezone.utc)
    LOGGER.info("Pipeline starting from %s", settings.project_root)
    LOGGER.info("Pipeline run id: %s", run_id)
    LOGGER.info("Kafka bootstrap servers: %s", settings.kafka_bootstrap_servers)
    LOGGER.info("Local lake: %s", settings.local_lake_dir)
    LOGGER.info("S3 enabled: %s, bucket: s3://%s/", settings.use_s3, settings.bucket_name)
    LOGGER.info("Datasets in pipeline: %s", [dataset.name for dataset in settings.datasets])

    if args.monitor_terminals:
        open_monitor_terminals(settings)

    if not args.skip_docker:
        with stage_log("docker", "Starting local Docker architecture"):
            start_architecture(settings, kafka_enabled=not args.skip_kafka)

    with stage_log("storage", "Preparing local lake and optional S3 bucket"):
        s3_info = prepare_storage(settings)
    if s3_info:
        bucket_name, bucket_region = s3_info
        if bucket_name != settings.bucket_name or bucket_region != settings.aws_region:
            settings = replace(
                settings,
                bucket_name=bucket_name,
                aws_region=bucket_region,
                athena_results_s3=_resolve_athena_results_s3(
                    settings.athena_results_s3,
                    settings.bucket_name,
                    bucket_name,
                ),
            )
            os.environ["AWS_BUCKET_NAME"] = bucket_name
            os.environ["AWS_DEFAULT_REGION"] = bucket_region
            os.environ["AWS_REGION"] = bucket_region
            _persist_s3_settings(settings.project_root, bucket_name, bucket_region)
            LOGGER.warning(
                "S3 bucket updated to %s in %s; syncing AWS settings for the pipeline.",
                bucket_name,
                bucket_region,
            )

    if not args.skip_kafka:
        with stage_log("kafka", "Ensuring Kafka topics exist"):
            pipeline["ensure_topics"](settings)
        with stage_log("kafka", "Producing CSV events into Kafka"):
            produced_counts = pipeline["produce_all"](settings)
            LOGGER.info("Kafka produced counts: %s", produced_counts)
        with stage_log("bronze", "Consuming Kafka events into Bronze JSONL"):
            if args.clean_bronze:
                clean_bronze(settings)
            else:
                warn_existing_bronze(settings)
            consumed_counts = pipeline["consume_to_bronze"](settings, produced_counts)
            LOGGER.info("Kafka consumed to Bronze counts: %s", consumed_counts)

    with stage_log("silver", "Building Spark session"):
        spark = pipeline["build_spark"]()
    try:
        with stage_log("silver", "Running Bronze to Silver transformations"):
            silver_outputs = pipeline["run_bronze_to_silver"](settings, spark)
            LOGGER.info("Silver outputs: %s", silver_outputs)
        run_gold = args.run_gold and not args.skip_gold
        if not run_gold:
            gold_outputs: dict[str, Path] = {}
            LOGGER.info("Skipping Silver to Gold transformations (use --run-gold to enable)")
        else:
            with stage_log("gold", "Running Silver to Gold transformations"):
                gold_outputs = pipeline["run_silver_to_gold"](settings, spark)
                LOGGER.info("Gold outputs: %s", gold_outputs)
    finally:
        LOGGER.info("Stopping Spark session")
        spark.stop()

    run_totals: dict[str, int] | None = None
    if not args.skip_metadata:
        lambda_role_arn = args.lambda_role_arn or os.getenv("AWS_LAMBDA_ROLE_ARN")
        auto_deploy_lambda = True
        with stage_log("metadata", "Registering metadata in MongoDB and PostgreSQL/RDS"):
            run_totals = register_metadata(
                settings,
                silver_outputs,
                gold_outputs,
                pipeline,
                auto_deploy_lambda=auto_deploy_lambda,
                lambda_role_arn=lambda_role_arn,
                run_id=run_id,
            )

    with stage_log("athena", "Registering S3 data lake tables in Glue/Athena"):
        athena_specs = pipeline["specs_from_outputs"](settings, "silver", silver_outputs)
        if gold_outputs:
            athena_specs.extend(pipeline["specs_from_outputs"](settings, "gold", gold_outputs))
        athena_tables = pipeline["register_parquet_tables"](settings, athena_specs)
        athena_results = pipeline["run_validation_queries"](settings, athena_tables)
        LOGGER.info("Athena validation query executions: %s", athena_results)

    if not args.skip_metadata:
        pipeline["register_pipeline_run"](
            settings,
            run_id,
            status="success",
            started_at=run_started_at,
            finished_at=datetime.now(timezone.utc),
            kafka_enabled=not args.skip_kafka,
            use_s3=settings.use_s3,
            max_rows_per_dataset=settings.max_rows_per_dataset,
            producer_batch_size=settings.producer_batch_size,
            producer_batch_delay_seconds=settings.producer_batch_delay_seconds,
            consumer_timeout_seconds=settings.consumer_timeout_seconds,
            datasets_count=(run_totals or {}).get("datasets_count"),
            total_row_count=(run_totals or {}).get("total_row_count"),
            total_size_bytes=(run_totals or {}).get("total_size_bytes"),
        )

    LOGGER.info("Pipeline completed.")
    LOGGER.info("Local lake: %s", settings.local_lake_dir)
    LOGGER.info("S3 bucket: s3://%s/", settings.bucket_name)


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / "pipeline.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    configure_module_file_logger("src.storage", LOG_DIR / STAGE_LOGS["s3"], formatter)
    configure_module_file_logger("src.aws.s3_client", LOG_DIR / STAGE_LOGS["s3"], formatter)


def configure_module_file_logger(logger_name: str, path: Path, formatter: logging.Formatter) -> None:
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@contextmanager
def stage_log(stage_name: str, message: str):
    log_file = LOG_DIR / STAGE_LOGS[stage_name]
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    started_at = time.monotonic()
    LOGGER.info("[START] %s", message)
    try:
        yield
    except Exception:
        LOGGER.exception("[FAIL] %s", message)
        raise
    else:
        LOGGER.info("[OK] %s in %.1fs", message, time.monotonic() - started_at)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def import_pipeline_dependencies(args: argparse.Namespace) -> dict[str, object]:
    pipeline: dict[str, object] = {}
    try:
        from src.spark.session import build_spark
        from src.spark.transforms import run_bronze_to_silver, run_silver_to_gold
        from src.aws.athena_catalog import register_parquet_tables, run_validation_queries, specs_from_outputs
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing dependency '{missing}'. Install project dependencies with: "
            f"python -m pip install -r requirements.txt"
        ) from exc

    pipeline.update(
        {
            "build_spark": build_spark,
            "run_bronze_to_silver": run_bronze_to_silver,
            "run_silver_to_gold": run_silver_to_gold,
            "register_parquet_tables": register_parquet_tables,
            "run_validation_queries": run_validation_queries,
            "specs_from_outputs": specs_from_outputs,
        }
    )

    if not args.skip_kafka:
        try:
            from src.kafka.admin import ensure_topics
            from src.kafka.consumer import consume_to_bronze
            from src.kafka.producer import produce_all
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            raise SystemExit(
                f"Missing Kafka dependency '{missing}'. Install project dependencies with: "
                f"python -m pip install -r requirements.txt"
            ) from exc
        pipeline.update(
            {
                "ensure_topics": ensure_topics,
                "consume_to_bronze": consume_to_bronze,
                "produce_all": produce_all,
            }
        )

    if not args.skip_metadata:
        try:
            from src.aws.lambda_client import build_lambda_event, ensure_s3_event_triggers, invoke_pipeline_lambda
            from src.aws.rds_client import register_column_stats, register_dataset_run, register_pipeline_run
            from src.mongo import record_layer_metadata
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            raise SystemExit(
                f"Missing metadata dependency '{missing}'. Install project dependencies with: "
                f"python -m pip install -r requirements.txt"
            ) from exc
        pipeline.update(
            {
                "build_lambda_event": build_lambda_event,
                "ensure_s3_event_triggers": ensure_s3_event_triggers,
                "invoke_pipeline_lambda": invoke_pipeline_lambda,
                "register_pipeline_run": register_pipeline_run,
                "register_dataset_run": register_dataset_run,
                "register_column_stats": register_column_stats,
                "record_layer_metadata": record_layer_metadata,
            }
        )

    return pipeline


def start_architecture(settings: Settings, kafka_enabled: bool = True) -> None:
    compose_file = settings.project_root / "docker" / "docker-compose.yml"
    run_command(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        cwd=settings.project_root,
        description="docker compose up -d",
    )
    if kafka_enabled:
        wait_for_docker_service(compose_file, settings.project_root, "zookeeper", "Zookeeper")
        try:
            wait_for_docker_service(compose_file, settings.project_root, "kafka", "Kafka", timeout_seconds=30)
        except TimeoutError:
            logs = docker_logs(compose_file, settings.project_root, "kafka", tail=120)
            if "KeeperErrorCode = NodeExists" in logs:
                LOGGER.warning(
                    "Kafka exited because Zookeeper still had a stale broker id. "
                    "Restarting Zookeeper and Kafka to clear the ephemeral broker registration."
                )
                run_command(
                    ["docker", "compose", "-f", str(compose_file), "restart", "zookeeper"],
                    cwd=settings.project_root,
                    description="restart zookeeper",
                )
                wait_for_docker_service(compose_file, settings.project_root, "zookeeper", "Zookeeper")
                run_command(
                    ["docker", "compose", "-f", str(compose_file), "up", "-d", "kafka"],
                    cwd=settings.project_root,
                    description="start kafka",
                )
                wait_for_docker_service(compose_file, settings.project_root, "kafka", "Kafka")
            else:
                LOGGER.error("Kafka logs before failure:\n%s", logs)
                raise
        wait_for_port("localhost", 9092, "Kafka")
    wait_for_port("localhost", 27017, "MongoDB")
    wait_for_port("localhost", 5432, "PostgreSQL/RDS")
    LOGGER.info("Docker architecture is reachable")


def prepare_storage(settings: Settings) -> tuple[str, str] | None:
    try:
        from src.storage import LakeWriter
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing storage dependency '{missing}'. Install project dependencies with: "
            f"python -m pip install -r requirements.txt"
        ) from exc
    try:
        return LakeWriter(settings).prepare()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_athena_results_s3(value: str, old_bucket: str, new_bucket: str) -> str:
    if not value or "<bucket>" in value:
        return f"s3://{new_bucket}/athena/results/"
    if value == f"s3://{old_bucket}/athena/results/":
        return f"s3://{new_bucket}/athena/results/"
    return value


def _persist_s3_settings(project_root: Path, bucket_name: str, bucket_region: str) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    saw_bucket = False
    saw_default_region = False
    saw_region = False
    updated: list[str] = []
    for line in lines:
        if line.startswith("AWS_BUCKET_NAME="):
            updated.append(f"AWS_BUCKET_NAME={bucket_name}")
            saw_bucket = True
        elif line.startswith("AWS_DEFAULT_REGION="):
            updated.append(f"AWS_DEFAULT_REGION={bucket_region}")
            saw_default_region = True
        elif line.startswith("AWS_REGION="):
            updated.append(f"AWS_REGION={bucket_region}")
            saw_region = True
        else:
            updated.append(line)
    if not saw_bucket:
        updated.append(f"AWS_BUCKET_NAME={bucket_name}")
    if not saw_default_region:
        updated.append(f"AWS_DEFAULT_REGION={bucket_region}")
    if not saw_region:
        updated.append(f"AWS_REGION={bucket_region}")
    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def clean_bronze(settings: Settings) -> None:
    bronze_dir = settings.local_lake_dir / "bronze"
    if not bronze_dir.exists():
        LOGGER.info("No existing Bronze directory to clean")
        return
    for dataset in settings.datasets:
        dataset_dir = bronze_dir / dataset.name
        if dataset_dir.exists():
            LOGGER.info("Cleaning existing Bronze files for %s at %s", dataset.name, dataset_dir)
            shutil.rmtree(dataset_dir)


def warn_existing_bronze(settings: Settings) -> None:
    for dataset in settings.datasets:
        dataset_dir = settings.local_lake_dir / "bronze" / dataset.name
        existing_files = sorted(dataset_dir.glob("*.jsonl")) if dataset_dir.exists() else []
        if existing_files:
            LOGGER.warning(
                "Bronze already has %s JSONL files for %s. Silver will process historical Bronze plus this run. "
                "Use --clean-bronze for a fresh run.",
                len(existing_files),
                dataset.name,
            )


def wait_for_port(host: str, port: int, service_name: str, timeout_seconds: int = 90) -> None:
    LOGGER.info("Waiting for %s on %s:%s", service_name, host, port)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex((host, port)) == 0:
                LOGGER.info("%s port is open on %s:%s", service_name, host, port)
                return
        time.sleep(2)
    raise TimeoutError(f"{service_name} did not become available on {host}:{port}")


def wait_for_docker_service(
    compose_file: Path,
    project_root: Path,
    service: str,
    service_name: str,
    timeout_seconds: int = 90,
) -> None:
    LOGGER.info("Waiting for Docker service %s to be running", service_name)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "ps", "--status", "running", "--services"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=True,
        )
        running_services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        if service in running_services:
            LOGGER.info("Docker service %s is running", service_name)
            return
        time.sleep(2)
    raise TimeoutError(f"{service_name} Docker service is not running after {timeout_seconds}s")


def docker_logs(compose_file: Path, project_root: Path, service: str, tail: int = 80) -> str:
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "logs", f"--tail={tail}", service],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout + result.stderr


def run_command(command: list[str], cwd: Path, description: str) -> None:
    LOGGER.info("Running: %s", description)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    for line in process.stdout:
        LOGGER.info("%s", line.rstrip())
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def open_monitor_terminals(settings: Settings) -> None:
    if os.name != "nt":
        LOGGER.warning("--monitor-terminals is only implemented for Windows PowerShell.")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in ["pipeline.log", *STAGE_LOGS.values()]:
        (LOG_DIR / file_name).touch(exist_ok=True)

    compose_file = settings.project_root / "docker" / "docker-compose.yml"
    monitors = {
        "Pipeline": f"Get-Content -Path '{LOG_DIR / 'pipeline.log'}' -Wait",
        "Kafka": f"Get-Content -Path '{LOG_DIR / STAGE_LOGS['kafka']}' -Wait",
        "Bronze": f"Get-Content -Path '{LOG_DIR / STAGE_LOGS['bronze']}' -Wait",
        "Silver": f"Get-Content -Path '{LOG_DIR / STAGE_LOGS['silver']}' -Wait",
        "Gold": f"Get-Content -Path '{LOG_DIR / STAGE_LOGS['gold']}' -Wait",
    }
    for title, command in monitors.items():
        subprocess.Popen(
            [
                "powershell",
                "-NoExit",
                "-Command",
                f"$host.UI.RawUI.WindowTitle = 'Pipeline - {title}'; Set-Location '{settings.project_root}'; {command}",
            ],
            cwd=settings.project_root,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    LOGGER.info("Opened monitor terminals for pipeline, Kafka, Bronze, Silver, and Gold logs")


def register_metadata(
    settings: Settings,
    silver_outputs: dict[str, Path],
    gold_outputs: dict[str, Path],
    pipeline: dict[str, object],
    *,
    auto_deploy_lambda: bool = False,
    lambda_role_arn: str | None = None,
    run_id: str | None = None,
) -> dict[str, int]:
    skip_lambda = False
    totals = {"datasets_count": 0, "total_row_count": 0, "total_size_bytes": 0}
    use_external_lambda = settings.use_s3
    if settings.use_s3:
        use_external_lambda = _ensure_lambda_s3_triggers(
            settings,
            pipeline,
            auto_deploy_lambda,
            lambda_role_arn,
        )
        if use_external_lambda:
            LOGGER.info("S3 notifications will trigger the audit Lambda for Bronze/Silver/Gold.")
        else:
            LOGGER.warning(
                "S3 triggers not configured; falling back to manual Lambda invocation."
            )

    for layer, outputs in (("silver", silver_outputs), ("gold", gold_outputs)):
        for dataset, path in outputs.items():
            location = str(path)
            if not use_external_lambda:
                lambda_event = pipeline["build_lambda_event"](layer=layer, dataset=dataset, location=location)
                LOGGER.info("Prepared Lambda trigger event: %s", lambda_event)
                if not skip_lambda:
                    try:
                        lambda_response = pipeline["invoke_pipeline_lambda"](settings, lambda_event)
                        LOGGER.info("Invoked AWS Lambda: %s", lambda_response)
                    except ClientError as exc:
                        if _is_lambda_missing(exc):
                            resolved = _handle_missing_lambda(
                                settings,
                                lambda_event,
                                pipeline,
                                auto_deploy_lambda,
                                lambda_role_arn,
                            )
                            if not resolved:
                                skip_lambda = True
                        else:
                            raise
            else:
                LOGGER.info("Skipping manual Lambda invocation for %s/%s; S3 triggers handle it.", layer, dataset)
            dataset_summary, column_stats, file_stats = _build_dataset_metadata(
                settings, layer, dataset, path, run_id
            )
            pipeline["record_layer_metadata"](
                settings,
                dataset,
                layer,
                location,
                run_id=run_id,
                metadata={"file_stats": file_stats},
            )
            pipeline["register_dataset_run"](
                settings,
                dataset,
                layer,
                location,
                run_id=run_id,
                summary=dataset_summary,
            )
            pipeline["register_column_stats"](
                settings,
                dataset,
                layer,
                run_id=run_id or "",
                row_count=dataset_summary.get("row_count", 0),
                column_stats=column_stats,
            )
            totals["datasets_count"] += 1
            totals["total_row_count"] += int(dataset_summary.get("row_count", 0) or 0)
            totals["total_size_bytes"] += int(dataset_summary.get("size_bytes", 0) or 0)
            LOGGER.info("Registered %s/%s metadata at %s", layer, dataset, location)
    return totals


def _is_lambda_missing(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException"


def _ensure_lambda_s3_triggers(
    settings: Settings,
    pipeline: dict[str, object],
    auto_deploy_lambda: bool,
    lambda_role_arn: str | None,
) -> bool:
    try:
        trigger_info = pipeline["ensure_s3_event_triggers"](settings)
        LOGGER.info("Configured S3 triggers for Lambda: %s", trigger_info)
        return True
    except (ClientError, ValueError) as exc:
        if _is_lambda_missing(exc):
            if not auto_deploy_lambda:
                return False
            role_arn = lambda_role_arn or os.getenv("AWS_LAMBDA_ROLE_ARN")
            if not role_arn:
                role_arn = _ensure_lambda_role(settings)
            if not role_arn:
                return False
            LOGGER.warning("Lambda missing; deploying before configuring S3 triggers.")
            try:
                _deploy_lambda(settings, role_arn)
            except (ClientError, subprocess.CalledProcessError) as deploy_exc:
                LOGGER.warning("Lambda deploy failed: %s", deploy_exc)
                return False
            try:
                trigger_info = pipeline["ensure_s3_event_triggers"](settings)
                LOGGER.info("Configured S3 triggers for Lambda after deploy: %s", trigger_info)
                return True
            except (ClientError, ValueError) as retry_exc:
                LOGGER.warning("Failed to configure S3 triggers after deploy: %s", retry_exc)
                return False
        LOGGER.warning("Failed to configure S3 triggers for Lambda: %s", exc)
        return False


def _handle_missing_lambda(
    settings: Settings,
    lambda_event: dict[str, str],
    pipeline: dict[str, object],
    auto_deploy_lambda: bool,
    lambda_role_arn: str | None,
) -> bool:
    role_arn = lambda_role_arn or os.getenv("AWS_LAMBDA_ROLE_ARN")
    if not role_arn and auto_deploy_lambda:
        role_arn = _ensure_lambda_role(settings)
    if not role_arn:
        LOGGER.warning(
            "Lambda %s not found and no role ARN available. "
            "Set AWS_LAMBDA_ROLE_ARN or AWS_LAMBDA_ROLE_NAME to enable auto-deploy.",
            settings.lambda_function_name,
        )
        return False
    if not auto_deploy_lambda:
        LOGGER.warning(
            "Lambda %s not found; skipping Lambda invocation.",
            settings.lambda_function_name,
        )
        return False
    LOGGER.warning("Lambda %s not found; deploying before retry.", settings.lambda_function_name)
    try:
        _deploy_lambda(settings, role_arn)
        lambda_response = pipeline["invoke_pipeline_lambda"](settings, lambda_event)
        LOGGER.info("Invoked AWS Lambda after deploy: %s", lambda_response)
        return True
    except (ClientError, subprocess.CalledProcessError) as exc:
        LOGGER.warning("Lambda deploy failed for %s: %s", settings.lambda_function_name, exc)
        return False


def _ensure_lambda_role(settings: Settings) -> str | None:
    role_names = _candidate_lambda_roles(settings)
    iam = boto3.client("iam")
    access_denied = False
    for role_name in role_names:
        try:
            response = iam.get_role(RoleName=role_name)
            return response["Role"]["Arn"]
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code == "NoSuchEntity":
                continue
            if code == "AccessDenied":
                access_denied = True
                break
            LOGGER.warning("Failed to read Lambda role %s: %s", role_name, exc)
            return None
        except (NoCredentialsError, PartialCredentialsError) as exc:
            LOGGER.warning("AWS credentials not available for role lookup: %s", exc)
            return None
    if access_denied:
        guessed = _guess_role_arn(role_names)
        if guessed:
            LOGGER.warning("Using guessed Lambda role ARN %s due to iam:GetRole access denied.", guessed)
            return guessed
        return None

    role_name = os.getenv("AWS_LAMBDA_ROLE_NAME") or f"{settings.lambda_function_name}-role"
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(_lambda_assume_policy()),
            Description="Auto-created role for pipeline Lambda audit",
        )
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
        _attach_lambda_audit_policy(iam, role_name, settings.bucket_name)
        time.sleep(5)
        return response["Role"]["Arn"]
    except (ClientError, NoCredentialsError, PartialCredentialsError) as exc:
        LOGGER.warning("Failed to create Lambda role %s: %s", role_name, exc)
        return None


def _candidate_lambda_roles(settings: Settings) -> list[str]:
    names = []
    env_name = os.getenv("AWS_LAMBDA_ROLE_NAME")
    if env_name:
        names.append(env_name)
    names.extend(["LabRole", "Lab Role", f"{settings.lambda_function_name}-role"])
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _guess_role_arn(role_names: list[str]) -> str | None:
    role_name = next((name for name in role_names if _role_name_for_arn(name)), None)
    if not role_name:
        return None
    try:
        account_id = boto3.client("sts").get_caller_identity()["Account"]
    except (ClientError, NoCredentialsError, PartialCredentialsError) as exc:
        LOGGER.warning("AWS credentials not available for role ARN guess: %s", exc)
        return None
    return f"arn:aws:iam::{account_id}:role/{role_name}"


def _role_name_for_arn(role_name: str) -> str | None:
    if not role_name or re.search(r"\s", role_name):
        return None
    return role_name


def _lambda_assume_policy() -> dict[str, object]:
    assume_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    return assume_policy


def _attach_lambda_audit_policy(iam, role_name: str, bucket_name: str) -> None:
    audit_prefix = "lambda/audit"
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket_name}/{audit_prefix}/*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="MusicRecommenderLambdaAuditWrite",
        PolicyDocument=json.dumps(policy),
    )


def _deploy_lambda(settings: Settings, role_arn: str) -> None:
    deploy_script = settings.project_root / "scripts" / "deploy_lambda.py"
    if not deploy_script.exists():
        raise FileNotFoundError(f"Lambda deploy script not found at {deploy_script}")
    run_command(
        [sys.executable, str(deploy_script), "--role-arn", role_arn],
        cwd=settings.project_root,
        description="deploy pipeline lambda",
    )


def _build_dataset_metadata(
    settings: Settings,
    layer: str,
    dataset: str,
    local_path: Path,
    run_id: str | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    profile = _collect_parquet_metadata(local_path)
    s3_uri = f"s3://{settings.bucket_name}/{layer}/{dataset}/" if settings.use_s3 else None
    row_count = profile.get("row_count") or 0
    file_count = profile.get("file_count") or 0
    size_bytes = profile.get("size_bytes") or 0
    summary = {
        "run_id": run_id,
        "s3_uri": s3_uri,
        "row_count": row_count,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "column_count": profile.get("column_count") or 0,
        "schema_hash": profile.get("schema_hash"),
        "sample_files": profile.get("sample_files"),
    }
    column_stats = profile.get("column_stats") or []
    file_stats = profile.get("file_stats") or []
    return summary, column_stats, file_stats


def _collect_parquet_metadata(path: Path) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        LOGGER.warning("pyarrow not installed; skipping parquet metadata collection")
        return {
            "row_count": 0,
            "file_count": 0,
            "size_bytes": 0,
            "columns": [],
            "column_count": 0,
            "null_counts": None,
            "sample_files": [],
        }

    files = _parquet_files(path)
    if not files:
        return {
            "row_count": 0,
            "file_count": 0,
            "size_bytes": 0,
            "columns": [],
            "column_count": 0,
            "schema_hash": None,
            "column_stats": [],
            "file_stats": [],
            "sample_files": [],
        }

    size_bytes = sum(file.stat().st_size for file in files)
    schema = pq.read_schema(files[0])
    columns = [
        {"name": field.name, "type": str(field.type)}
        for field in schema
        if not field.name.startswith("__")
    ]
    column_stats: dict[str, dict[str, object]] = {
        col["name"]: {
            "name": col["name"],
            "type": col["type"],
            "null_count": 0,
            "min": None,
            "max": None,
        }
        for col in columns
    }
    file_stats: list[dict[str, object]] = []
    row_count = 0
    for file in files:
        parquet = pq.ParquetFile(file)
        metadata = parquet.metadata
        if metadata is None:
            continue
        row_count += metadata.num_rows
        file_stats.append(
            {
                "file_name": file.name,
                "file_path": str(file),
                "size_bytes": file.stat().st_size,
                "row_count": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
            }
        )
        for rg_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_index)
            for col_index in range(row_group.num_columns):
                column = row_group.column(col_index)
                stats = column.statistics
                if not stats:
                    continue
                column_name = schema.names[col_index]
                if column_name.startswith("__"):
                    continue
                stat = column_stats.get(column_name)
                if not stat:
                    continue
                if stats.has_null_count:
                    stat["null_count"] = int(stat.get("null_count", 0)) + int(stats.null_count)
                if stats.has_min_max:
                    stat["min"] = _min_stat(stat.get("min"), stats.min)
                    stat["max"] = _max_stat(stat.get("max"), stats.max)

    return {
        "row_count": row_count,
        "file_count": len(files),
        "size_bytes": size_bytes,
        "columns": columns,
        "column_count": len(columns),
        "schema_hash": _schema_hash(columns),
        "column_stats": [_stringify_column_stats(stat) for stat in column_stats.values()],
        "file_stats": file_stats,
        "sample_files": [str(file.name) for file in files[:5]],
    }


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    return []


def _schema_hash(columns: list[dict[str, str]]) -> str | None:
    if not columns:
        return None
    payload = json.dumps(columns, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _min_stat(current: Any, candidate: Any) -> Any:
    if current is None:
        return candidate
    if candidate is None:
        return current
    return candidate if candidate < current else current


def _max_stat(current: Any, candidate: Any) -> Any:
    if current is None:
        return candidate
    if candidate is None:
        return current
    return candidate if candidate > current else current


def _stringify_column_stats(stat: dict[str, object]) -> dict[str, object]:
    return {
        "name": stat.get("name"),
        "type": stat.get("type"),
        "null_count": stat.get("null_count"),
        "min": _stringify_stat(stat.get("min")),
        "max": _stringify_stat(stat.get("max")),
    }


def _stringify_stat(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    main()
