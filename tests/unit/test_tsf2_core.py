"""Unit tests for tsf2_core shared helpers."""

from __future__ import annotations

from datetime import date

from tsf2_core.biweek import get_biweek_for_date
from tsf2_core.constants import MARKER, biweek_data_prefix
from tsf2_core.s3 import marker_exists
from tsf2_core.timeseries import family_encode
from tests.conftest import FakeS3Client


def test_family_encode():
    assert family_encode("GROCERY I") == "GROCERY_I"
    assert family_encode("BREAD/BAKERY") == "BREAD_BAKERY"


def test_biweek_data_prefix():
    assert biweek_data_prefix("processed/example/", 2024, 5) == "processed/example/2024/BW-5/"


def test_marker_exists_with_client():
    prefix = "processed/sarimax-subprime/biweekly/2024/BW-5/"
    client = FakeS3Client({f"{prefix}{MARKER}"})
    assert marker_exists(client, "bucket", prefix) is True
    assert marker_exists(FakeS3Client(), "bucket", prefix) is False


def test_get_biweek_for_date_matches_shared_calendar():
    period = get_biweek_for_date(date(2024, 3, 10))
    assert period is not None
    assert period.biweek_num == 5
