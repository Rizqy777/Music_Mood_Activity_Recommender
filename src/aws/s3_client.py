from __future__ import annotations

import logging
from pathlib import Path
import re
import uuid

import boto3
from botocore.exceptions import ClientError

from src.config import Settings

LOGGER = logging.getLogger(__name__)


class S3DataLake:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bucket_name = settings.bucket_name
        self.bucket_region = settings.aws_region
        self.client = self._build_client(self.bucket_region)

    def _build_client(self, region: str | None):
        return boto3.client(
            "s3",
            region_name=region,
            endpoint_url=self.settings.s3_endpoint_url,
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            LOGGER.info("S3 bucket exists and is accessible: %s", self.bucket_name)
        except ClientError as exc:
            if self._maybe_redirect_for_bucket(exc):
                return
            if self._is_missing_bucket(exc):
                self._create_or_select_bucket()
                return
            if self._is_bad_request(exc):
                LOGGER.warning(
                    "HeadBucket returned Bad Request for %s; attempting to create the bucket.",
                    self.bucket_name,
                )
                self._create_or_select_bucket()
                return
            if self._is_forbidden(exc):
                LOGGER.warning(
                    "S3 bucket %s is forbidden; selecting a new bucket name in this account.",
                    self.bucket_name,
                )
                self._create_alternate_bucket()
                return
            raise

    def _create_or_select_bucket(self) -> None:
        try:
            self._create_bucket(self.settings.bucket_name)
            self.bucket_name = self.settings.bucket_name
            LOGGER.info("Created S3 bucket: %s", self.bucket_name)
        except ClientError as exc:
            if self._is_bucket_owned_by_you(exc):
                self.bucket_name = self.settings.bucket_name
                LOGGER.info("S3 bucket already owned by you: %s", self.bucket_name)
                return
            if self._is_bucket_already_exists(exc):
                self._create_alternate_bucket()
                return
            redirect_region = self._extract_region_from_error(exc)
            if redirect_region and redirect_region != self.bucket_region:
                self._switch_region(redirect_region)
                self._create_bucket(self.settings.bucket_name)
                self.bucket_name = self.settings.bucket_name
                LOGGER.info("Created S3 bucket in %s: %s", redirect_region, self.bucket_name)
                return
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

    def _create_bucket(self, bucket_name: str) -> None:
        region = self.bucket_region or self.client.meta.region_name
        try:
            self.client.create_bucket(**self._create_bucket_args(bucket_name, region))
        except ClientError as exc:
            redirect_region = self._extract_region_from_error(exc)
            if redirect_region and redirect_region != region:
                self._switch_region(redirect_region)
                self.client.create_bucket(**self._create_bucket_args(bucket_name, redirect_region))
                return
            raise

    def _create_bucket_args(self, bucket_name: str, region: str | None) -> dict[str, object]:
        args: dict[str, object] = {"Bucket": bucket_name}
        if region and region != "us-east-1":
            args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        return args

    def _create_alternate_bucket(self) -> None:
        candidates = self._candidate_bucket_names()
        for name in candidates:
            try:
                self._create_bucket(name)
                self.bucket_name = name
                LOGGER.warning("Using fallback S3 bucket: %s", self.bucket_name)
                return
            except ClientError as exc:
                if self._is_bucket_owned_by_you(exc):
                    self.bucket_name = name
                    LOGGER.info("S3 bucket already owned by you: %s", self.bucket_name)
                    return
                if self._is_bucket_already_exists(exc):
                    continue
                if self._is_access_denied(exc) or self._is_forbidden(exc):
                    break
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
                "S3 bucket name is not accessible and no fallback bucket is available. "
                "Set AWS_BUCKET_NAME to a bucket this role can access."
            )
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

    def _candidate_bucket_names(self) -> list[str]:
        base = self._sanitize_bucket_name(self.settings.bucket_name)
        account_id = self._account_id()
        names: list[str] = []
        if account_id:
            names.append(self._sanitize_bucket_name(f"{base}-{account_id}"))
            names.append(self._sanitize_bucket_name(f"music-recommender-data-lake-{account_id}"))
        names.append(self._sanitize_bucket_name(f"{base}-{uuid.uuid4().hex[:6]}"))
        # Deduplicate while preserving order
        unique: list[str] = []
        seen = set()
        for name in names:
            if name and name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _account_id(self) -> str | None:
        try:
            sts = boto3.client("sts", region_name=self.bucket_region)
            return str(sts.get_caller_identity()["Account"])
        except ClientError:
            return None

    def _maybe_redirect_for_bucket(self, exc: ClientError) -> bool:
        region = self._extract_region_from_error(exc)
        if not region or region == self.bucket_region:
            return False
        self._switch_region(region)
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            return False
        LOGGER.info("S3 bucket resolved in region %s: %s", region, self.bucket_name)
        return True

    def _switch_region(self, region: str) -> None:
        self.bucket_region = region
        self.client = self._build_client(region)

    @staticmethod
    def _is_missing_bucket(exc: ClientError) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"404", "NoSuchBucket", "NotFound"} or status == 404

    @staticmethod
    def _is_bad_request(exc: ClientError) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status == 400 or code in {"BadRequest", "InvalidBucketName", "AuthorizationHeaderMalformed"}

    @staticmethod
    def _is_forbidden(exc: ClientError) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status == 403 or code in {"403", "AccessDenied", "Forbidden"}

    @staticmethod
    def _extract_region_from_error(exc: ClientError) -> str | None:
        headers = exc.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        region = headers.get("x-amz-bucket-region")
        if region:
            return str(region)
        region = exc.response.get("Error", {}).get("Region")
        if region:
            return str(region)
        message = str(exc.response.get("Error", {}).get("Message", ""))
        match = re.search(r"([a-z]{2}-[a-z-]+-\d)", message)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _is_access_denied(exc: ClientError) -> bool:
        return str(exc.response.get("Error", {}).get("Code", "")) == "AccessDenied"

    @staticmethod
    def _is_bucket_already_exists(exc: ClientError) -> bool:
        return str(exc.response.get("Error", {}).get("Code", "")) == "BucketAlreadyExists"

    @staticmethod
    def _is_bucket_owned_by_you(exc: ClientError) -> bool:
        return str(exc.response.get("Error", {}).get("Code", "")) == "BucketAlreadyOwnedByYou"

    @staticmethod
    def _sanitize_bucket_name(value: str) -> str:
        name = value.strip().lower()
        name = re.sub(r"[^a-z0-9.-]", "-", name)
        name = re.sub(r"\.{2,}", ".", name)
        name = name.strip(".-")
        if len(name) < 3:
            name = f"{name}---".strip(".-")
        if len(name) > 63:
            name = name[:63].rstrip(".-")
        return name
