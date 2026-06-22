"""Biweek calendar helpers for the TSF2 daily-ingestion pipeline."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from botocore.exceptions import ClientError

from tsf2_core.constants import MARKER, REQUIRED_DAILY_DATASETS, SUBPRIME_BIWEEKLY_PREFIX

# Daily ingestion begins after historical bootstrap; BW-13 is the first live biweek.
CUTOFF_DATE = date(2017, 7, 1)
FIRST_BIWEEK = 13
FIRST_HALF_END_DAY = 15

SUBPRIME_OUTPUT_PREFIX = SUBPRIME_BIWEEKLY_PREFIX


@dataclass(frozen=True)
class BiweekPeriod:
    date: date
    year: int
    biweek_num: int
    biweek_start: date
    biweek_end: date

    def to_payload(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "year": f"{self.year}",
            "biweek_num": f"{self.biweek_num}",
            "biweek_start": self.biweek_start.isoformat(),
            "biweek_end": self.biweek_end.isoformat(),
        }

    @property
    def deduplication_id(self) -> str:
        return f"{self.year}-BW-{self.biweek_num}"


def parse_event_date(event: dict) -> Optional[date]:
    """Extract the upload date from an EventBridge S3 Object Created event."""
    detail = event.get("detail") or {}
    obj = detail.get("object") or {}
    key = obj.get("key")
    if not key:
        return None

    parts = key.split("/")
    # raw/daily/YYYY/MM/DD/<dataset>.csv
    if len(parts) < 6 or parts[0] != "raw" or parts[1] != "daily":
        return None

    try:
        return date(int(parts[2]), int(parts[3]), int(parts[4]))
    except (TypeError, ValueError):
        return None


def get_biweek_for_date(value: date) -> Optional[BiweekPeriod]:
    """Map a calendar date to its biweek period.

    Each month has two biweeks: days 1-15 and days 16 through month-end.
    """
    if value < CUTOFF_DATE:
        return None

    if value.day <= FIRST_HALF_END_DAY:
        biweek_num = (value.month - 1) * 2 + 1
        biweek_start = date(value.year, value.month, 1)
        biweek_end = date(value.year, value.month, FIRST_HALF_END_DAY)
    else:
        biweek_num = (value.month - 1) * 2 + 2
        biweek_start = date(value.year, value.month, FIRST_HALF_END_DAY + 1)
        _, last_day = calendar.monthrange(value.year, value.month)
        biweek_end = date(value.year, value.month, last_day)

    if value.year == CUTOFF_DATE.year and biweek_num < FIRST_BIWEEK:
        return None

    return BiweekPeriod(
        date=value,
        year=value.year,
        biweek_num=biweek_num,
        biweek_start=biweek_start,
        biweek_end=biweek_end,
    )


def _daily_prefix(day: date) -> str:
    return f"raw/daily/{day.year}/{day.month:02d}/{day.day:02d}/"


def biweek_data_is_complete(s3_client, bucket_name: str, period: BiweekPeriod) -> bool:
    """Return True when every required daily dataset exists for each day in the biweek."""
    current = period.biweek_start
    while current <= period.biweek_end:
        prefix = _daily_prefix(current)
        for dataset in REQUIRED_DAILY_DATASETS:
            try:
                s3_client.head_object(Bucket=bucket_name, Key=f"{prefix}{dataset}.csv")
            except ClientError:
                return False
        current += timedelta(days=1)
    return True


def biweek_already_processed(s3_client, bucket_name: str, period: BiweekPeriod) -> bool:
    """Return True when the subprime output marker already exists for this biweek."""
    prefix = f"{SUBPRIME_OUTPUT_PREFIX}{period.year}/BW-{period.biweek_num}/{MARKER}"
    try:
        s3_client.head_object(Bucket=bucket_name, Key=prefix)
        return True
    except ClientError:
        return False
