from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_files() -> None:
    for env_file, override in (
        (PROJECT_ROOT / ".env", False),
        (PROJECT_ROOT / "aws_credentials.env", True),
    ):
        if load_dotenv:
            load_dotenv(env_file, override=override)
        else:
            _load_env_file(env_file, override=override)
    _normalize_aws_env()


def _load_env_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if override or key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _normalize_aws_env() -> None:
    aliases = {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",
        "aws_default_region": "AWS_DEFAULT_REGION",
        "aws_bucket_name": "AWS_BUCKET_NAME",
        "AWS_BUCKET": "AWS_BUCKET_NAME",
        "S3_ENDPOINT_URL": "AWS_S3_ENDPOINT_URL",
    }
    for source, target in aliases.items():
        if source in os.environ and target not in os.environ:
            os.environ[target] = os.environ[source]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    topic: str
    id_column: str


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    local_lake_dir: Path
    bucket_name: str
    aws_region: str
    s3_endpoint_url: str | None
    use_s3: bool
    kafka_bootstrap_servers: str
    kafka_group_id: str
    max_rows_per_dataset: int | None
    producer_batch_size: int
    producer_batch_delay_seconds: float
    consumer_timeout_seconds: int
    mongo_uri: str
    mongo_database: str
    rds_dsn: str
    producer_batch_delay_seconds: float

    @property
    def mood_dataset(self) -> DatasetConfig:
        return DatasetConfig(
            name="mood_dataset",
            path=self.data_dir / "mood_dataset.csv",
            topic=os.getenv("KAFKA_MOOD_TOPIC", "mood_dataset_events"),
            id_column="uri",
        )

    @property
    def tracks_dataset(self) -> DatasetConfig:
        return DatasetConfig(
            name="tracks_dataset",
            path=self.data_dir / "spotify_tracks_dataset.csv",
            topic=os.getenv("KAFKA_TRACKS_TOPIC", "tracks_dataset_events"),
            id_column="track_id",
        )

    @property
    def datasets(self) -> Iterable[DatasetConfig]:
        return (self.mood_dataset, self.tracks_dataset)


def load_settings() -> Settings:
    _load_env_files()
    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=Path(os.getenv("DATASETS_DIR", PROJECT_ROOT / "datasets")).resolve(),
        local_lake_dir=Path(os.getenv("LOCAL_LAKE_DIR", PROJECT_ROOT / "data_lake")).resolve(),
        bucket_name=os.getenv("AWS_BUCKET_NAME") or _default_bucket_name(),
        aws_region=os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1",
        s3_endpoint_url=os.getenv("AWS_S3_ENDPOINT_URL") or None,
        use_s3=_env_bool("USE_S3", True),
        kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_group_id=os.getenv("KAFKA_GROUP_ID", "music-recommender-pipeline"),
        max_rows_per_dataset=_env_int("MAX_ROWS_PER_DATASET", None),
        producer_batch_size=_env_int("PRODUCER_BATCH_SIZE", 25) or 500,
        producer_batch_delay_seconds=_env_float("PRODUCER_BATCH_DELAY_SECONDS", 0.5),
        consumer_timeout_seconds=_env_int("CONSUMER_TIMEOUT_SECONDS", 20) or 20,
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        mongo_database=os.getenv("MONGO_DATABASE", "music_recommender"),
        rds_dsn=os.getenv(
            "RDS_DSN",
            "dbname=music_recommender user=postgres password=postgres host=localhost port=5432",
        ),
    )


def _default_bucket_name() -> str:
    account_id = os.getenv("AWS_ACCOUNT_ID")
    if account_id:
        return f"music-recommender-data-lake-{account_id}"
    try:
        import boto3

        account_id = boto3.client("sts").get_caller_identity()["Account"]
        return f"music-recommender-data-lake-{account_id}"
    except Exception:
        return "music-recommender-data-lake"
