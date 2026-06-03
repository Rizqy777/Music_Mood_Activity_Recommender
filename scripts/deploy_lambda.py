from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.aws.lambda_client import ensure_s3_event_triggers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy or update the pipeline audit AWS Lambda.")
    parser.add_argument(
        "--role-arn",
        required=True,
        help="IAM role ARN with Lambda basic execution and S3 PutObject permissions.",
    )
    return parser.parse_args()


def build_zip() -> bytes:
    source = ROOT / "src" / "aws" / "lambda_handler.py"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(source, "lambda_handler.py")
    return buffer.getvalue()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    client = boto3.client("lambda", region_name=settings.aws_region)
    code = build_zip()
    env = {
        "Variables": {
            "AUDIT_BUCKET": settings.bucket_name,
            "AUDIT_PREFIX": "lambda/audit",
        }
    }
    try:
        response = client.create_function(
            FunctionName=settings.lambda_function_name,
            Runtime="python3.11",
            Role=args.role_arn,
            Handler="lambda_handler.handler",
            Code={"ZipFile": code},
            Description="Pipeline audit Lambda for Music Mood Activity Recommender",
            Timeout=30,
            MemorySize=128,
            Environment=env,
            Publish=True,
        )
        print("Lambda creada:", response["FunctionArn"])
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code")) != "ResourceConflictException":
            raise
        client.update_function_code(
            FunctionName=settings.lambda_function_name,
            ZipFile=code,
            Publish=True,
        )
        client.update_function_configuration(
            FunctionName=settings.lambda_function_name,
            Runtime="python3.11",
            Role=args.role_arn,
            Handler="lambda_handler.handler",
            Timeout=30,
            MemorySize=128,
            Environment=env,
        )
        print("Lambda actualizada:", settings.lambda_function_name)

    trigger_info = ensure_s3_event_triggers(settings)
    print("S3 triggers configurados:", trigger_info)


if __name__ == "__main__":
    main()
