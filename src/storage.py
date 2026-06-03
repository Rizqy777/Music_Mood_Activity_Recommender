from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.aws.s3_client import S3DataLake
from src.config import Settings

LOGGER = logging.getLogger(__name__)


class LakeWriter:
    """Writes a local lake copy and optionally mirrors it to S3."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.s3 = S3DataLake(settings) if settings.use_s3 else None

    def prepare(self) -> tuple[str, str] | None:
        self.settings.local_lake_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Local lake directory ready: %s", self.settings.local_lake_dir)
        if self.s3:
            LOGGER.info("Ensuring S3 bucket is available: %s", self.settings.bucket_name)
            self.s3.ensure_bucket()
            LOGGER.info("S3 bucket ready: %s", self.s3.bucket_name)
            return self.s3.bucket_name, self.s3.bucket_region or self.settings.aws_region
        else:
            LOGGER.info("S3 mirroring disabled")
        return None

    def local_dataset_dir(self, layer: str, dataset: str) -> Path:
        path = self.settings.local_lake_dir / layer / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def s3_prefix(self, layer: str, dataset: str) -> str:
        return f"{layer}/{dataset}"

    def s3_uri(self, layer: str, dataset: str) -> str:
        bucket_name = self.s3.bucket_name if self.s3 else self.settings.bucket_name
        return f"s3://{bucket_name}/{self.s3_prefix(layer, dataset)}/"

    def append_jsonl(self, layer: str, dataset: str, rows: list[dict[str, Any]], file_name: str) -> Path:
        target = self.local_dataset_dir(layer, dataset) / file_name
        with target.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        if self.s3:
            key = f"{self.s3_prefix(layer, dataset)}/{file_name}"
            self.s3.upload_file(target, key)
            LOGGER.info("Uploaded %s to %s/%s", target, self.s3.bucket_name, key)
        return target

    def mirror_directory_to_s3(self, layer: str, dataset: str) -> None:
        if not self.s3:
            return
        LOGGER.info("Mirroring %s/%s directory to S3", layer, dataset)
        self.s3.upload_directory(
            self.local_dataset_dir(layer, dataset),
            self.s3_prefix(layer, dataset),
        )
        LOGGER.info("Finished mirroring %s/%s directory to S3", layer, dataset)
