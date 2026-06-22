"""Unit tests for biweek calendar and S3 completeness helpers."""

from __future__ import annotations

from datetime import date

import pytest

from shared.biweek import (
    CUTOFF_DATE,
    FIRST_BIWEEK,
    BiweekPeriod,
    biweek_already_processed,
    biweek_data_is_complete,
    get_biweek_for_date,
    parse_event_date,
)

from tests.conftest import FakeS3Client, make_s3_event


def test_parse_event_date_extracts_upload_date():
    event = make_s3_event("raw/daily/2024/03/10/train.csv")
    assert parse_event_date(event) == date(2024, 3, 10)


@pytest.mark.parametrize(
    "key",
    [
        "processed/sarimax-subprime/biweekly/2024/BW-5/_COMPLETE",
        "raw/daily/2024/03",
        "raw/weekly/2024/03/10/train.csv",
        "raw/daily/not/a/date/train.csv",
    ],
)
def test_parse_event_date_rejects_invalid_keys(key):
    assert parse_event_date(make_s3_event(key)) is None


def test_get_biweek_for_date_before_cutoff_returns_none():
    assert get_biweek_for_date(date(2017, 6, 30)) is None


def test_get_biweek_for_date_before_first_live_biweek_returns_none():
    assert get_biweek_for_date(date(2017, 6, 15)) is None


def test_get_biweek_for_date_first_valid_biweek():
    period = get_biweek_for_date(CUTOFF_DATE)
    assert period == BiweekPeriod(
        date=CUTOFF_DATE,
        year=2017,
        biweek_num=FIRST_BIWEEK,
        biweek_start=date(2017, 7, 1),
        biweek_end=date(2017, 7, 15),
    )


def test_get_biweek_for_date_first_half_of_month():
    period = get_biweek_for_date(date(2024, 2, 10))
    assert period is not None
    assert period.biweek_num == 3
    assert period.biweek_start == date(2024, 2, 1)
    assert period.biweek_end == date(2024, 2, 15)


def test_get_biweek_for_date_second_half_of_month():
    period = get_biweek_for_date(date(2024, 2, 20))
    assert period is not None
    assert period.biweek_num == 4
    assert period.biweek_start == date(2024, 2, 16)
    assert period.biweek_end == date(2024, 2, 29)


def test_biweek_period_payload_and_deduplication_id():
    period = get_biweek_for_date(date(2024, 3, 10))
    assert period is not None
    assert period.to_payload() == {
        "date": "2024-03-10",
        "year": "2024",
        "biweek_num": "5",
        "biweek_start": "2024-03-01",
        "biweek_end": "2024-03-15",
    }
    assert period.deduplication_id == "2024-BW-5"


def test_biweek_data_is_complete_when_all_daily_files_exist(july_2017_first_half_period, fake_s3_complete):
    assert biweek_data_is_complete(fake_s3_complete, "bucket", july_2017_first_half_period) is True


def test_biweek_data_is_complete_when_a_dataset_is_missing(july_2017_first_half_period, fake_s3_complete):
    missing_key = (
        f"raw/daily/{july_2017_first_half_period.biweek_start.year}/"
        f"{july_2017_first_half_period.biweek_start.month:02d}/"
        f"{july_2017_first_half_period.biweek_start.day:02d}/oil.csv"
    )
    fake_s3_complete.existing_keys.remove(missing_key)
    assert biweek_data_is_complete(fake_s3_complete, "bucket", july_2017_first_half_period) is False


def test_biweek_already_processed_when_marker_exists(july_2017_first_half_period):
    marker_key = (
        f"processed/sarimax-subprime/biweekly/{july_2017_first_half_period.year}/"
        f"BW-{july_2017_first_half_period.biweek_num}/_COMPLETE"
    )
    s3_client = FakeS3Client({marker_key})
    assert biweek_already_processed(s3_client, "bucket", july_2017_first_half_period) is True


def test_biweek_already_processed_when_marker_missing(july_2017_first_half_period):
    assert biweek_already_processed(FakeS3Client(), "bucket", july_2017_first_half_period) is False
