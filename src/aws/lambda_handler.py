from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone

import boto3


def handler(event, context):
    """AWS Lambda entrypoint for pipeline notifications."""
    bucket = os.environ.get("AUDIT_BUCKET") or os.environ.get("AWS_BUCKET_NAME")
    prefix = os.environ.get("AUDIT_PREFIX", "lambda/audit")
    if isinstance(event, dict) and event.get("Records"):
        return _handle_s3_event(event, context, bucket, prefix)

    payload = _build_audit_payload(event, context)
    if bucket:
        payload["audit_s3_uri"] = _write_audit_payload(bucket, prefix, payload)
    return {
        "statusCode": 200,
        "body": json.dumps(payload, ensure_ascii=False),
    }


def _handle_s3_event(event, context, audit_bucket: str | None, prefix: str) -> dict[str, object]:
    records = []
    for record in event.get("Records", []):
        if record.get("eventSource") != "aws:s3":
            continue
        bucket = record.get("s3", {}).get("bucket", {}).get("name")
        key = record.get("s3", {}).get("object", {}).get("key")
        if not bucket or not key:
            continue
        key = urllib.parse.unquote_plus(key)
        layer, dataset = _parse_layer_dataset(key)
        derived_event = {
            "event_type": "s3_object_created",
            "layer": layer,
            "dataset": dataset,
            "location": f"s3://{bucket}/{key}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": record.get("eventName"),
        }
        payload = _build_audit_payload(derived_event, context)
        target_bucket = audit_bucket or bucket
        if target_bucket:
            payload["audit_s3_uri"] = _write_audit_payload(target_bucket, prefix, payload)
        records.append(payload)

    return {
        "statusCode": 200,
        "body": json.dumps({"records": records}, ensure_ascii=False),
    }


def _parse_layer_dataset(key: str) -> tuple[str, str]:
    parts = key.split("/")
    if len(parts) >= 2:
        return parts[0] or "unknown", parts[1] or "unknown"
    return "unknown", "unknown"


def _build_audit_payload(event, context) -> dict[str, object]:
    return {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "aws_request_id": getattr(context, "aws_request_id", None),
    }


def _write_audit_payload(bucket: str, prefix: str, payload: dict[str, object]) -> str:
    event = payload.get("event") or {}
    key = (
        f"{prefix.rstrip('/')}/"
        f"{event.get('layer', 'unknown')}/"
        f"{event.get('dataset', 'unknown')}/"
        f"{payload['received_at'].replace(':', '-')}.json"
    )
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"
