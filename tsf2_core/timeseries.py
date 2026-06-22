"""Time series aggregation and S3 loading helpers."""

from __future__ import annotations

import json
from io import BytesIO

import pandas as pd

from tsf2_core.constants import (
    EXOG_FEATURES,
    FAMILIES_MAPPING_KEY,
    NON_TWO_YEAR_FAMILIES,
    NON_TWO_YEAR_STORES,
    PERIOD_MAP,
    TIMESERIES_FAMILY_HISTORICAL_PREFIX,
    TIMESERIES_STORE_HISTORICAL_PREFIX,
    TWO_YEAR_CUTOFF,
    TWO_YEAR_FAMILIES,
    TWO_YEAR_STORES,
)


def family_encode(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def build_time_series(data: pd.DataFrame) -> tuple[dict, dict]:
    """Aggregate prime data into per-family and per-store daily time series."""
    print("\n[TIMESERIES] Building aggregated time series per family and store")

    ts_per_family: dict = {}
    ts_per_store: dict = {}

    print("  Building time series per family...")
    for family in data["family"].unique():
        if family in NON_TWO_YEAR_FAMILIES:
            ts_per_family[family] = data.loc[
                (data["date"] > PERIOD_MAP[NON_TWO_YEAR_FAMILIES[family]])
                & (data["family"] == family)
            ].groupby(["date"]).agg(EXOG_FEATURES)
        elif family in TWO_YEAR_FAMILIES:
            ts_per_family[family] = data.loc[
                (data["date"] > TWO_YEAR_CUTOFF) & (data["family"] == family)
            ].groupby(["date"]).agg(EXOG_FEATURES)
    print(f"    ✓ Built time series for {len(ts_per_family)} families")

    print("  Building time series per store...")
    for store in range(1, 55):
        if store in NON_TWO_YEAR_STORES:
            store_data = data.loc[
                (data["date"] > PERIOD_MAP[NON_TWO_YEAR_STORES[store]])
                & (data["store_nbr"] == store)
            ].groupby(["date"]).agg(EXOG_FEATURES)
        elif store in TWO_YEAR_STORES:
            store_data = data.loc[
                (data["date"] > TWO_YEAR_CUTOFF) & (data["store_nbr"] == store)
            ].groupby(["date"]).agg(EXOG_FEATURES)
        else:
            continue
        if len(store_data) > 0:
            ts_per_store[store] = store_data
    print(f"    ✓ Built time series for {len(ts_per_store)} stores")

    return ts_per_family, ts_per_store


def build_and_upload_time_series(s3_client, bucket_name: str, processed_data: pd.DataFrame) -> bool:
    """Build time series aggregations and upload parquet files to S3."""
    ts_per_family, ts_per_store = build_time_series(processed_data)

    print(f"\n[S3] Uploading family time series to s3://{bucket_name}/{TIMESERIES_FAMILY_HISTORICAL_PREFIX}")
    for family, ts_data in ts_per_family.items():
        try:
            parquet_buffer = BytesIO()
            ts_data.to_parquet(parquet_buffer, index=True)
            parquet_buffer.seek(0)
            s3_key = f"{TIMESERIES_FAMILY_HISTORICAL_PREFIX}{family_encode(family)}.parquet"
            s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=parquet_buffer.getvalue())
        except Exception as exc:
            print(f"    ✗ Error uploading family '{family}': {exc}")
            return False
    print(f"  ✓ Uploaded {len(ts_per_family)} family time series")

    print(f"\n[S3] Uploading store time series to s3://{bucket_name}/{TIMESERIES_STORE_HISTORICAL_PREFIX}")
    for store, ts_data in ts_per_store.items():
        try:
            parquet_buffer = BytesIO()
            ts_data.to_parquet(parquet_buffer, index=True)
            parquet_buffer.seek(0)
            s3_key = f"{TIMESERIES_STORE_HISTORICAL_PREFIX}store_{store:02d}.parquet"
            s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=parquet_buffer.getvalue())
        except Exception as exc:
            print(f"    ✗ Error uploading store {store}: {exc}")
            return False
    print(f"  ✓ Uploaded {len(ts_per_store)} store time series")
    return True


def load_time_series(s3_client, bucket_name: str):
    """Load historical family/store time series parquet files from S3."""
    print("\n[TIMESERIES] Loading time series from S3")

    ts_per_family: dict = {}
    ts_per_store: dict = {}

    try:
        print("  Loading family name mapping...")
        response = s3_client.get_object(Bucket=bucket_name, Key=FAMILIES_MAPPING_KEY)
        family_mapping = json.loads(response["Body"].read().decode("utf-8"))
    except Exception as exc:
        print(f"    Warning: Could not load family mapping: {exc}. Aborting.")
        return None, None

    print("  Loading family time series...")
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=TIMESERIES_FAMILY_HISTORICAL_PREFIX):
        for obj in page.get("Contents") or []:
            if not obj["Key"].endswith(".parquet"):
                continue
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=obj["Key"])
                ts_data = pd.read_parquet(BytesIO(response["Body"].read()))
                encoded_name = obj["Key"].split("/")[-1].replace(".parquet", "")
                family_name = family_mapping[encoded_name]
                ts_per_family[family_name] = ts_data
            except KeyError:
                print(f"  ✗ Encoded family name '{encoded_name}' not found in mapping. Aborting.")
                return None, None
            except Exception as exc:
                print(f"  ✗ Could not load family time series from {obj['Key']}: {exc}")
                return None, None
    print(f"    ✓ Loaded {len(ts_per_family)} family time series")

    print("  Loading store time series...")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=TIMESERIES_STORE_HISTORICAL_PREFIX):
        for obj in page.get("Contents") or []:
            if not obj["Key"].endswith(".parquet"):
                continue
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=obj["Key"])
                ts_data = pd.read_parquet(BytesIO(response["Body"].read()))
                filename = obj["Key"].split("/")[-1].replace(".parquet", "")
                if not filename.startswith("store_"):
                    print(f"  ✗ Unexpected store time series filename '{filename}'. Aborting.")
                    return None, None
                store_num = int(filename.split("_")[1])
                ts_per_store[store_num] = ts_data
            except Exception as exc:
                print(f"    Warning: Could not load store time series from {obj['Key']}: {exc}")
                return None, None
    print(f"    ✓ Loaded {len(ts_per_store)} store time series")
    return ts_per_family, ts_per_store
