"""S3 and CloudFormation helper utilities."""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from tsf2_core.constants import MARKER


def get_stack_output(env_name: str, export_name: str) -> str:
    cf = boto3.client("cloudformation")
    response = cf.describe_stacks(StackName=f"{env_name}-StorageStack")
    outputs = response["Stacks"][0]["Outputs"]
    try:
        return next(o["OutputValue"] for o in outputs if o["ExportName"] == export_name)
    except StopIteration as exc:
        raise RuntimeError(f"Failed to get stack output '{export_name}'") from exc


def marker_exists(s3_client, bucket: str, prefix: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}{MARKER}")
        return True
    except ClientError:
        return False


def write_marker(s3_client, bucket: str, prefix: str) -> None:
    s3_client.put_object(Bucket=bucket, Key=f"{prefix}{MARKER}", Body=b"")
