#!/usr/bin/env python3
"""
Process historical data: load from S3, apply SARIMAX Prime transformations,
save processed parquet files back to S3, and clean up local files.
"""

import os
import sys
import shutil
import json
import tempfile
from pathlib import Path
from io import BytesIO

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from bootstrap import get_stack_output, marker_exists, write_marker
from tsf2_core.constants import (
    DATASET_NAMES,
    MARKER,
    NON_TWO_YEAR_FAMILIES,
    PRIME_HISTORICAL_PREFIX,
    RAW_HISTORICAL_PREFIX,
    SUBPRIME_HISTORICAL_PREFIX,
    TWO_YEAR_FAMILIES,
)
from tsf2_core.timeseries import build_and_upload_time_series, family_encode
from tsf2_core.transforms import apply_subprime_transformations, fit_prime_transform

TEMP_DIR = Path(tempfile.mkdtemp(prefix="sarimax_prime_"))


def download_from_s3(s3_client, bucket_name):
    """Download historical datasets from S3."""
    print(f"\n[S3] Downloading datasets from s3://{bucket_name}/{RAW_HISTORICAL_PREFIX}")

    datasets = {}
    missing_files = []

    for dataset_name in DATASET_NAMES:
        s3_key = f"{RAW_HISTORICAL_PREFIX}{dataset_name}.csv"
        local_path = TEMP_DIR / f"{dataset_name}.csv"

        try:
            print(f"  Downloading {dataset_name}...", end=" ")
            s3_client.download_file(bucket_name, s3_key, str(local_path))

            if dataset_name == "stores":
                datasets[dataset_name] = pd.read_csv(local_path)
            else:
                datasets[dataset_name] = pd.read_csv(local_path, parse_dates=["date"])

            print(f"✓ ({len(datasets[dataset_name])} rows)")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                missing_files.append(s3_key)
                print("✗ (not found in S3)")
            else:
                print("✗")
                print(f"  Error downloading {s3_key}: {error_code}")
                return None
        except Exception as e:
            print("✗")
            print(f"  Error loading {dataset_name}: {e}")
            return None

    if missing_files:
        print(f"\n  Warning: {len(missing_files)} dataset(s) not found:")
        for key in missing_files:
            print(f"    - {key}")
        if not datasets:
            print("  Error: No datasets could be loaded from S3")
            return None

    print(f"\n✓ Downloaded {len(datasets)} dataset(s)")
    return datasets


def upload_subprime_to_s3(s3_client, bucket_name, subprime_data):
    print(f"\n[S3] Uploading subprime data to s3://{bucket_name}/{SUBPRIME_HISTORICAL_PREFIX}")

    try:
        parquet_buffer = BytesIO()
        subprime_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        s3_key = f"{SUBPRIME_HISTORICAL_PREFIX}data.parquet"
        print(f"  Uploading {s3_key}...", end=" ")
        s3_client.upload_fileobj(parquet_buffer, bucket_name, s3_key)
        print("✓")
        return True
    except ClientError as e:
        print("✗")
        print(f"  Error uploading to S3: {e}")
        return False


def upload_to_s3(s3_client, bucket_name, processed_data, lambdas, hmvs):
    print(f"\n[S3] Uploading processed data to s3://{bucket_name}/{PRIME_HISTORICAL_PREFIX}")

    def save_json_to_s3(data, s3_key):
        buffer = BytesIO()
        buffer.write(json.dumps(data, indent=2).encode("utf-8"))
        buffer.seek(0)
        print(f"  Uploading {s3_key}...", end=" ")
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        print("✓")

    try:
        parquet_buffer = BytesIO()
        processed_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        s3_key = f"{PRIME_HISTORICAL_PREFIX}data.parquet"
        print(f"  Uploading {s3_key}...", end=" ")
        s3_client.upload_fileobj(parquet_buffer, bucket_name, s3_key)
        print("✓")

        save_json_to_s3(lambdas, f"{PRIME_HISTORICAL_PREFIX}lambdas.json")
        save_json_to_s3(hmvs, f"{PRIME_HISTORICAL_PREFIX}hmvs.json")
        family_mapping = {
            family_encode(k): k for k in (NON_TWO_YEAR_FAMILIES.keys() | TWO_YEAR_FAMILIES)
        }
        save_json_to_s3(family_mapping, f"{PRIME_HISTORICAL_PREFIX}families_mapping.json")
        return True
    except ClientError as e:
        print("✗")
        print(f"  Error uploading to S3: {e}")
        return False


def cleanup_temp_files():
    if TEMP_DIR.exists():
        print(f"\n[CLEANUP] Removing temporary directory: {TEMP_DIR}")
        try:
            shutil.rmtree(TEMP_DIR)
            print("✓ Cleaned up temporary files")
        except Exception as e:
            print(f"✗ Error removing {TEMP_DIR}: {e}")
            return False
    return True


def main(env_name):
    s3_bucket_name = get_stack_output(env_name, f"{env_name}-DataBucketName")

    print("=" * 60)
    print("SARIMAX Prime Data Processing")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"S3 Bucket: {s3_bucket_name}")
    print(f"Temp Directory: {TEMP_DIR}")
    print()

    try:
        s3_client = boto3.client("s3")

        print("[1/9] Checking for historical data marker...")
        if not marker_exists(s3_bucket_name, RAW_HISTORICAL_PREFIX):
            print(
                f"✗ Error: Marker file not found at s3://{s3_bucket_name}/{RAW_HISTORICAL_PREFIX}{MARKER}"
            )
            print("  Please run: python bootstrap/1_raw.py {}".format(env_name))
            sys.exit(1)
        print("✓ Marker found. Raw data is ready for processing.")

        print("\n[2/9] Checking if processing has already been completed...")
        if marker_exists(s3_bucket_name, PRIME_HISTORICAL_PREFIX):
            print("✗ Error: Processing workflow has already been completed.")
            sys.exit(0)
        print("✓ No marker found. Ready to proceed with processing.")

        print("\n[3/9] Downloading historical data from S3...")
        datasets = download_from_s3(s3_client, s3_bucket_name)
        if datasets is None:
            print("Error: Failed to download datasets from S3")
            sys.exit(1)

        print("\n[4/9] Applying SARIMAX subprime transformations...")
        subprime_data = apply_subprime_transformations(datasets)

        print("\n[5/9] Uploading subprime data to S3...")
        if not upload_subprime_to_s3(s3_client, s3_bucket_name, subprime_data):
            print("Error: Failed to upload subprime data to S3")
            sys.exit(1)

        print("\n[6/9] Applying SARIMAX prime transformations...")
        processed_data, lambdas, hmvs = fit_prime_transform(subprime_data)

        print("\n[7/9] Uploading prime data to S3...")
        if not upload_to_s3(s3_client, s3_bucket_name, processed_data, lambdas, hmvs):
            print("Error: Failed to upload processed data to S3")
            sys.exit(1)

        print("\n[8/9] Building and uploading time series...")
        if not build_and_upload_time_series(s3_client, s3_bucket_name, processed_data):
            print("Error: Failed to build and upload time series")
            sys.exit(1)

        print("\n[9/9] Finalizing...")
        write_marker(s3_bucket_name, PRIME_HISTORICAL_PREFIX)
        cleanup_temp_files()

        print("\n" + "=" * 60)
        print("✓ Process completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        cleanup_temp_files()
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        cleanup_temp_files()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 2_sarimax_prime.py <env_name>")
        sys.exit(1)
    main(sys.argv[1])
