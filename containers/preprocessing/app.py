import os
import json
from enum import Enum
from io import BytesIO
from datetime import datetime, timedelta

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from tsf2_core.constants import (
    PRIME_BIWEEKLY_PREFIX,
    RAW_HISTORICAL_PREFIX,
    REQUIRED_DAILY_DATASETS,
    SUBPRIME_BIWEEKLY_PREFIX,
    biweek_data_prefix,
)
from tsf2_core.s3 import marker_exists, write_marker
from tsf2_core.transforms import apply_subprime_transformations


class IO(Enum):
    INPUT = 1
    OUTPUT = 2


def get_full_biweekly_prefix(year, biweek_num, io_type):
    if io_type == IO.INPUT:
        return biweek_data_prefix(PRIME_BIWEEKLY_PREFIX, year, biweek_num)
    return biweek_data_prefix(SUBPRIME_BIWEEKLY_PREFIX, year, biweek_num)


def load_daily_csvs(s3_client, bucket_name, biweek_start, biweek_end):
    """Load and concatenate CSV files from all days in biweek period."""
    print(f"[S3] Loading daily CSVs from {biweek_start.date()} to {biweek_end.date()}")

    current_date = biweek_start
    datasets = {name: [] for name in REQUIRED_DAILY_DATASETS}

    while current_date <= biweek_end:
        date_str = f"{current_date.year}/{current_date.month:02d}/{current_date.day:02d}"
        folder_prefix = f"raw/daily/{date_str}/"

        for dataset_name in REQUIRED_DAILY_DATASETS:
            s3_key = f"{folder_prefix}{dataset_name}.csv"
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                df = pd.read_csv(BytesIO(response["Body"].read()))
                if dataset_name != "stores" and "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                datasets[dataset_name].append(df)
                print(f"  ✓ Loaded {dataset_name} from {date_str}")
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchKey":
                    print(f"  ⚠ Error loading {dataset_name} from {date_str}: {e}")

        current_date += timedelta(days=1)

    concatenated = {}
    for dataset_name, dfs in datasets.items():
        if dfs:
            concatenated[dataset_name] = pd.concat(dfs, ignore_index=True)
            if dataset_name != "stores":
                concatenated[dataset_name] = concatenated[dataset_name].drop_duplicates()
            print(f"  ✓ Concatenated {dataset_name}: {len(concatenated[dataset_name])} rows")
        else:
            print(f"  ✗ No data found for {dataset_name}")
            return None

    return concatenated


def load_stores_csv(s3_client, bucket_name):
    s3_key = f"{RAW_HISTORICAL_PREFIX}stores.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_csv(BytesIO(response["Body"].read()))
        print(f"[S3] Loaded stores.csv with {len(df)} rows")
        return df
    except ClientError as e:
        print(f"Error loading stores.csv: {e}")
        return None


def upload_biweekly_data(s3_client, bucket_name, processed_data, year, biweek_num):
    try:
        s3_prefix = get_full_biweekly_prefix(year, biweek_num, IO.OUTPUT)
        print(f"\n[S3] Uploading processed data to s3://{bucket_name}/{s3_prefix}")

        parquet_buffer = BytesIO()
        processed_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        s3_key = f"{s3_prefix}data.parquet"
        print("  Uploading data.parquet...", end=" ")
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=parquet_buffer.getvalue())
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Error uploading to S3: {e}")
        return False


def lambda_handler(event, context):
    env_name = os.environ.get("ENVIRONMENT", "dev")

    try:
        print("=" * 70)
        print("Biweekly SARIMAX Subprime Data Processing Lambda")
        print("=" * 70)

        s3_client = boto3.client("s3")
        bucket_name = os.environ.get("DATA_BUCKET")
        if not bucket_name:
            return {"statusCode": 400, "body": json.dumps({"error": "DATA_BUCKET not set"})}

        print(f"Environment: {env_name}")
        print(f"Bucket: {bucket_name}\n")

        print("[1/7] Extracting parameters from Step Functions payload...")
        date_str = event.get("date")
        year = event.get("year")
        biweek_num = event.get("biweek_num")
        biweek_start_str = event.get("biweek_start")
        biweek_end_str = event.get("biweek_end")

        if not all([date_str, year is not None, biweek_num is not None, biweek_start_str, biweek_end_str]):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required payload parameters"}),
            }

        process_date = datetime.strptime(date_str, "%Y-%m-%d")
        biweek_start = datetime.strptime(biweek_start_str, "%Y-%m-%d")
        biweek_end = datetime.strptime(biweek_end_str, "%Y-%m-%d")
        print(f"✓ Date: {process_date.date()}")
        print(f"✓ Year: {year}, Biweek: {biweek_num}")
        print(f"✓ Period: {biweek_start.date()} to {biweek_end.date()}")

        print("\n[2/7] Checking if processing has already been completed...")
        if marker_exists(s3_client, bucket_name, get_full_biweekly_prefix(year, biweek_num, IO.OUTPUT)):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Processing workflow has already been completed."}),
            }
        print("✓ No marker found. Ready to proceed with processing.")

        print("\n[3/7] Loading daily CSVs from biweek period...")
        datasets = load_daily_csvs(s3_client, bucket_name, biweek_start, biweek_end)
        if datasets is None:
            return {"statusCode": 400, "body": json.dumps({"error": "Failed to load daily CSVs"})}

        print("\n[4/7] Loading stores.csv for feature engineering...")
        stores = load_stores_csv(s3_client, bucket_name)
        if stores is None:
            return {"statusCode": 400, "body": json.dumps({"error": "Failed to load stores.csv"})}
        datasets["stores"] = stores

        print("\n[5/7] Processing data with subprime transformations...")
        processed_data = apply_subprime_transformations(datasets)

        print("\n[6/7] Uploading processed data to S3...")
        if not upload_biweekly_data(s3_client, bucket_name, processed_data, year, biweek_num):
            return {"statusCode": 500, "body": json.dumps({"error": "Failed to upload processed data"})}

        print("\n[7/7] Finalizing...")
        write_marker(s3_client, bucket_name, get_full_biweekly_prefix(year, biweek_num, IO.OUTPUT))

        print("\n" + "=" * 70)
        print("✓ Processing completed successfully!")
        print("=" * 70)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Biweekly processing completed",
                    "year": year,
                    "biweek": biweek_num,
                    "period_start": biweek_start.date().isoformat(),
                    "period_end": biweek_end.date().isoformat(),
                    "rows_processed": len(processed_data),
                }
            ),
        }

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
