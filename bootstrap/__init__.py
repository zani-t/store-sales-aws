"""Backward-compatible bootstrap helpers; prefer tsf2_core imports in new code."""

import boto3

from tsf2_core.constants import MARKER, SIGNIFICANT_EXOG
from tsf2_core.s3 import (
    get_stack_output,
    marker_exists as _marker_exists,
    write_marker as _write_marker,
)
from tsf2_core.timeseries import family_encode, load_time_series

s3 = boto3.client("s3")

__all__ = [
    "MARKER",
    "SIGNIFICANT_EXOG",
    "family_encode",
    "get_stack_output",
    "load_time_series",
    "marker_exists",
    "write_marker",
]


def marker_exists(bucket, prefix):
    return _marker_exists(s3, bucket, prefix)


def write_marker(bucket, prefix):
    return _write_marker(s3, bucket, prefix)
