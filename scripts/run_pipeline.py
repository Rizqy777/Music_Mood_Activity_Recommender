from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the music recommender data pipeline.")
    parser.add_argument("--skip-docker", action="store_true", help="Do not start local Docker services.")
    parser.add_argument("--skip-kafka", action="store_true", help="Skip Kafka ingestion and only run Spark layers.")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip MongoDB/RDS metadata registration.")
    parser.add_argument("--skip-gold", action="store_true", help="Skip Gold transformations.")
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
    LOGGER.info("Pipeline starting from %s", settings.project_root)
    LOGGER.info("Kafka bootstrap servers: %s", settings.kafka_bootstrap_servers)
    LOGGER.info("Local lake: %s", settings.local_lake_dir)
    LOGGER.info("S3 enabled: %s, bucket: s3://%s/", settings.use_s3, settings.bucket_name)

    if args.monitor_terminals:
        open_monitor_terminals(settings)

    if not args.skip_docker:
        with stage_log("docker", "Starting local Docker architecture"):
            start_architecture(settings, kafka_enabled=not args.skip_kafka)

    with stage_log("storage", "Preparing local lake and optional S3 bucket"):
        prepare_storage(settings)

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

    if not args.skip_metadata:
        with stage_log("metadata", "Registering metadata in MongoDB and PostgreSQL/RDS"):
            register_metadata(settings, silver_outputs, gold_outputs, pipeline)

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
            from src.aws.lambda_stub import build_lambda_event
            from src.aws.rds_client import register_pipeline_output
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
                "register_pipeline_output": register_pipeline_output,
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


def prepare_storage(settings: Settings) -> None:
    try:
        from src.storage import LakeWriter
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing storage dependency '{missing}'. Install project dependencies with: "
            f"python -m pip install -r requirements.txt"
        ) from exc
    try:
        LakeWriter(settings).prepare()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


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
) -> None:
    for layer, outputs in (("silver", silver_outputs), ("gold", gold_outputs)):
        for dataset, path in outputs.items():
            location = str(path)
            lambda_event = pipeline["build_lambda_event"](layer=layer, dataset=dataset, location=location)
            LOGGER.info("Prepared Lambda trigger event: %s", lambda_event)
            pipeline["record_layer_metadata"](settings, dataset, layer, location)
            pipeline["register_pipeline_output"](settings, dataset, layer, location)
            LOGGER.info("Registered %s/%s metadata at %s", layer, dataset, location)


if __name__ == "__main__":
    main()
