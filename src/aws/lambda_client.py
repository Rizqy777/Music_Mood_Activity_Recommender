from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError, WaiterError

from src.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LambdaTriggerEvent:
    event_type: str
    layer: str
    dataset: str
    location: str
    created_at: str


S3_TRIGGER_RULES = (
    {"prefix": "bronze/", "suffix": ".jsonl"},
    {"prefix": "silver/", "suffix": "_SUCCESS"},
    {"prefix": "gold/", "suffix": "_SUCCESS"},
)


def build_lambda_event(layer: str, dataset: str, location: str) -> dict[str, str]:
    """Build the event sent to AWS Lambda after a data lake layer is written."""
    event = LambdaTriggerEvent(
        event_type="data_lake_layer_written",
        layer=layer,
        dataset=dataset,
        location=location,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return {
        "event_type": event.event_type,
        "layer": event.layer,
        "dataset": event.dataset,
        "location": event.location,
        "created_at": event.created_at,
    }


def invoke_pipeline_lambda(settings: Settings, event: dict[str, str]) -> dict[str, object]:
    """Invoke the configured AWS Lambda function using boto3."""
    client = boto3.client(
        "lambda",
        region_name=settings.aws_region,
    )
    try:
        response = client.invoke(
            FunctionName=settings.lambda_function_name,
            InvocationType=settings.lambda_invocation_type,
            Payload=json.dumps(event, ensure_ascii=False).encode("utf-8"),
        )
    except ClientError as exc:
        if _is_lambda_pending(exc):
            _wait_for_lambda_active(client, settings.lambda_function_name)
            response = client.invoke(
                FunctionName=settings.lambda_function_name,
                InvocationType=settings.lambda_invocation_type,
                Payload=json.dumps(event, ensure_ascii=False).encode("utf-8"),
            )
        else:
            raise
    return {
        "function_name": settings.lambda_function_name,
        "status_code": response.get("StatusCode"),
        "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
        "invocation_type": settings.lambda_invocation_type,
    }


def ensure_s3_event_triggers(settings: Settings) -> dict[str, object]:
    """Ensure S3 notifications are configured to trigger the audit Lambda."""
    lambda_client = boto3.client("lambda", region_name=settings.aws_region)
    s3_client = boto3.client("s3", region_name=settings.aws_region)
    _assert_bucket_region(s3_client, settings.bucket_name, settings.aws_region)
    function = lambda_client.get_function(FunctionName=settings.lambda_function_name)
    _wait_for_lambda_active(lambda_client, settings.lambda_function_name)
    function_arn = function["Configuration"]["FunctionArn"]
    _ensure_s3_invoke_permission(lambda_client, settings)

    current = s3_client.get_bucket_notification_configuration(Bucket=settings.bucket_name)
    desired_ids = {
        _notification_id(settings.lambda_function_name, rule["prefix"], rule["suffix"])
        for rule in S3_TRIGGER_RULES
    }
    existing = current.get("LambdaFunctionConfigurations", [])
    preserved = [config for config in existing if config.get("Id") not in desired_ids]
    new_configs = preserved + [
        _build_s3_notification_config(
            function_arn,
            settings.lambda_function_name,
            rule["prefix"],
            rule["suffix"],
        )
        for rule in S3_TRIGGER_RULES
    ]

    notification: dict[str, object] = {"LambdaFunctionConfigurations": new_configs}
    if "TopicConfigurations" in current:
        notification["TopicConfigurations"] = current["TopicConfigurations"]
    if "QueueConfigurations" in current:
        notification["QueueConfigurations"] = current["QueueConfigurations"]
    if "EventBridgeConfiguration" in current:
        notification["EventBridgeConfiguration"] = current["EventBridgeConfiguration"]

    last_error: ClientError | None = None
    for attempt in range(1, 6):
        try:
            s3_client.put_bucket_notification_configuration(
                Bucket=settings.bucket_name,
                NotificationConfiguration=notification,
            )
            last_error = None
            break
        except ClientError as exc:
            last_error = exc
            if _is_invalid_s3_destination(exc) and attempt < 5:
                time.sleep(2 * attempt)
                _wait_for_lambda_active(lambda_client, settings.lambda_function_name)
                continue
            raise

    if last_error is not None:
        raise last_error

    return {
        "function_arn": function_arn,
        "bucket_name": settings.bucket_name,
        "lambda_triggers": len(new_configs),
        "configured": True,
    }


def _ensure_s3_invoke_permission(lambda_client, settings: Settings) -> None:
    statement_id = f"{settings.lambda_function_name}-s3-invoke"
    try:
        lambda_client.add_permission(
            FunctionName=settings.lambda_function_name,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{settings.bucket_name}",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceConflictException":
            raise


def _build_s3_notification_config(
    function_arn: str,
    function_name: str,
    prefix: str,
    suffix: str,
) -> dict[str, object]:
    rules = []
    if prefix:
        rules.append({"Name": "prefix", "Value": prefix})
    if suffix:
        rules.append({"Name": "suffix", "Value": suffix})
    config: dict[str, object] = {
        "Id": _notification_id(function_name, prefix, suffix),
        "LambdaFunctionArn": function_arn,
        "Events": ["s3:ObjectCreated:*"],
    }
    if rules:
        config["Filter"] = {"Key": {"FilterRules": rules}}
    return config


def _notification_id(function_name: str, prefix: str, suffix: str) -> str:
    prefix_id = prefix.strip("/").replace("/", "-") or "root"
    suffix_id = suffix.strip("/").replace("/", "-").replace(".", "") or "all"
    return f"{function_name}-{prefix_id}-{suffix_id}"


def _wait_for_lambda_active(lambda_client, function_name: str) -> None:
    waiter = lambda_client.get_waiter("function_active")
    try:
        waiter.wait(FunctionName=function_name, WaiterConfig={"Delay": 2, "MaxAttempts": 15})
    except WaiterError as exc:
        raise ValueError(f"Lambda {function_name} did not become active: {exc}") from exc


def _assert_bucket_region(s3_client, bucket_name: str, lambda_region: str) -> None:
    try:
        response = s3_client.get_bucket_location(Bucket=bucket_name)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 403 or code in {"AccessDenied", "Forbidden"}:
            LOGGER.warning(
                "Access denied reading bucket location for %s; continuing without region validation.",
                bucket_name,
            )
            return
        raise
    bucket_region = _normalize_s3_region(response.get("LocationConstraint"))
    if bucket_region != lambda_region:
        raise ValueError(
            f"S3 bucket {bucket_name} is in {bucket_region}, but Lambda is in {lambda_region}. "
            "S3 notifications require the Lambda to be in the same region."
        )


def _normalize_s3_region(region: str | None) -> str:
    if not region:
        return "us-east-1"
    if region == "EU":
        return "eu-west-1"
    return region


def _is_invalid_s3_destination(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") == "InvalidArgument"


def _is_lambda_pending(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") == "ResourceConflictException"
