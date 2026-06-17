"""Shared pytest fixtures and import path setup."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = ROOT / "lambdas"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAMBDAS))

os.environ.setdefault("STATE_MACHINE_ARN", "arn:aws:states:us-east-1:123456789012:stateMachine:test")
os.environ.setdefault("QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue.fifo")
os.environ.setdefault("LOCK_TABLE_NAME", "test-orchestration-lock")
os.environ.setdefault("DATA_BUCKET", "test-data-bucket")
os.environ.setdefault("STARTER_FUNCTION_NAME", "test-starter")
os.environ.setdefault("MESSAGE_GROUP_ID", "tsf2-pipeline")


class FakeS3Client:
    """Minimal S3 client stub backed by an in-memory key set."""

    def __init__(self, existing_keys: set[str] | None = None) -> None:
        self.existing_keys = existing_keys or set()

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.existing_keys:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        return {}


def _daily_keys_for_period(period) -> set[str]:
    from datetime import timedelta

    from shared.biweek import REQUIRED_DAILY_DATASETS

    keys: set[str] = set()
    current = period.biweek_start
    while current <= period.biweek_end:
        prefix = f"raw/daily/{current.year}/{current.month:02d}/{current.day:02d}/"
        for dataset in REQUIRED_DAILY_DATASETS:
            keys.add(f"{prefix}{dataset}.csv")
        current += timedelta(days=1)
    return keys


@pytest.fixture
def july_2017_first_half_period():
    from shared.biweek import get_biweek_for_date

    period = get_biweek_for_date(date(2017, 7, 10))
    assert period is not None
    return period


@pytest.fixture
def fake_s3_complete(july_2017_first_half_period):
    return FakeS3Client(_daily_keys_for_period(july_2017_first_half_period))


def make_s3_event(key: str) -> dict:
    return {
        "detail": {
            "bucket": {"name": "test-data-bucket"},
            "object": {"key": key},
        }
    }
