from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from src.config import Settings

LOGGER = logging.getLogger(__name__)


class S3DataLake:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bucket_name = settings.bucket_name
        self.client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            endpoint_url=settings.s3_endpoint_url,
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            LOGGER.info("S3 bucket exists and is accessible: %s", self.bucket_name)
        except ClientError as exc:
            if self._is_missing_bucket(exc):
                self._create_or_select_bucket()
                return
            raise

    def _create_or_select_bucket(self) -> None:
        try:
            create_args: dict[str, object] = {"Bucket": self.settings.bucket_name}
            if self.settings.aws_region != "us-east-1":
                create_args["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.settings.aws_region
                }
            self.client.create_bucket(**create_args)
            self.bucket_name = self.settings.bucket_name
            LOGGER.info("Created S3 bucket: %s", self.bucket_name)
        except ClientError as exc:
            if not self._is_access_denied(exc):
                raise
            try:
                fallback_bucket = self._first_available_bucket()
            except ClientError as list_exc:
                if self._is_access_denied(list_exc):
                    raise RuntimeError(
                        "AWS denied automatic S3 setup: the role cannot create buckets "
                        "and cannot list existing buckets. Grant s3:CreateBucket or set "
                        "AWS_BUCKET_NAME to a bucket the role can access."
                    ) from list_exc
                raise
            if not fallback_bucket:
                raise RuntimeError(
                    "S3 bucket does not exist and the current AWS role cannot create buckets. "
                    "Set AWS_BUCKET_NAME to an existing lab bucket in aws_credentials.env."
                ) from exc
            self.bucket_name = fallback_bucket
            self.client.head_bucket(Bucket=self.bucket_name)
            LOGGER.info("Using first accessible S3 bucket: %s", self.bucket_name)

    def upload_file(self, source: Path, key: str) -> None:
        self.client.upload_file(str(source), self.bucket_name, key)
        LOGGER.info("S3 upload complete: s3://%s/%s", self.bucket_name, key)

    def upload_directory(self, source_dir: Path, prefix: str) -> None:
        for path in source_dir.rglob("*"):
            if path.is_file():
                relative = path.relative_to(source_dir).as_posix()
                self.upload_file(path, f"{prefix.rstrip('/')}/{relative}")

    def _first_available_bucket(self) -> str | None:
        response = self.client.list_buckets()
        buckets = response.get("Buckets", [])
        if not buckets:
            return None
        return str(buckets[0]["Name"])

    @staticmethod
    def _is_missing_bucket(exc: ClientError) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"404", "NoSuchBucket", "NotFound"} or status == 404

    @staticmethod
    def _is_access_denied(exc: ClientError) -> bool:
        return str(exc.response.get("Error", {}).get("Code", "")) == "AccessDenied"
