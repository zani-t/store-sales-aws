#!/usr/bin/env python3
"""
Train SARIMAX models per family and per store on biweekly data.
Downloads subprime data from S3, applies prime transformations, trains models,
saves them to S3. Models are stored in <model_bucket>/sarimax/biweekly/<year>/BW-<biweek>/.
"""

import warnings
import tempfile
import datetime
import uuid
from pathlib import Path
from io import BytesIO
from datetime import datetime as dt
from decimal import Decimal

import boto3
import pandas as pd
import pickle
import shutil
import os
import sys
import json
from botocore.exceptions import ClientError
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

from tsf2_core.constants import (
    FAMILIES_MAPPING_KEY,
    MARKER,
    PRIME_BIWEEKLY_PREFIX,
    SARIMAX_MODEL_BIWEEKLY_PREFIX,
    SIGNIFICANT_EXOG,
    SUBPRIME_BIWEEKLY_PREFIX,
    SUBPRIME_HISTORICAL_PREFIX,
)
from tsf2_core.s3 import marker_exists, write_marker
from tsf2_core.timeseries import build_time_series, family_encode
from tsf2_core.transforms import fit_prime_transform

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SUBPRIME_INPUT_PREFIX = SUBPRIME_BIWEEKLY_PREFIX
PRIME_OUTPUT_PREFIX = PRIME_BIWEEKLY_PREFIX
SARIMAX_OUTPUT_PREFIX = SARIMAX_MODEL_BIWEEKLY_PREFIX

TEMP_DIR = Path(tempfile.mkdtemp(prefix="sarimax_biweekly_"))
FAMILY_DIR = TEMP_DIR / "family"
STORE_DIR = TEMP_DIR / "store"


def ensure_directories_exist():
    """Create temporary output directories."""
    print(f"\n[SETUP] Creating temporary directories")
    
    for directory in [FAMILY_DIR, STORE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def load_subprime_data(s3_client, bucket_name, target_year, target_biweek):
    """Load historical and biweekly subprime data from S3 up to target biweek.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        target_year: Target year for data loading
        target_biweek: Target biweek number for data loading
    
    Returns:
        tuple: (combined_data DataFrame, target_year, target_biweek)
    """
    try:
        all_data = []
        
        # Step 1: Load historical data
        print(f"\n[S3] Loading historical subprime data...")
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=f"{SUBPRIME_HISTORICAL_PREFIX}data.parquet"
            )
            hist_data = pd.read_parquet(BytesIO(response['Body'].read()))
            all_data.append(hist_data)
            print(f"  ✓ Loaded historical data: {len(hist_data)} rows")
        except ClientError as e:
            print(f"  ⚠ Could not load historical data: {e}")
        
        # Step 2: Discover biweekly folders up to target biweek
        print(f"\n[S3] Discovering biweekly folders (target: {target_year}/BW-{target_biweek})...")
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Get all year folders
        year_folders = []
        pages = paginator.paginate(Bucket=bucket_name, Prefix=SUBPRIME_INPUT_PREFIX, Delimiter='/')
        for page in pages:
            for prefix in page.get('CommonPrefixes', []):
                year_folders.append(prefix['Prefix'])
        
        if not year_folders:
            print(f"  ⚠ No biweekly year folders found")
        else:
            print(f"  Found {len(year_folders)} year folder(s)")
        
        # Step 3: Load biweekly data up to target biweek
        biweek_count = 0
        for year_folder in year_folders:
            # Extract year from path
            year = int(year_folder.rstrip('/').split('/')[-1])
            
            # Skip years after target year
            if year > target_year:
                continue
            
            # Get all biweek folders for this year
            pages = paginator.paginate(Bucket=bucket_name, Prefix=year_folder, Delimiter='/')
            for page in pages:
                for prefix in page.get('CommonPrefixes', []):
                    biweek_folder = prefix['Prefix']
                    biweek_str = biweek_folder.rstrip('/').split('/')[-1]
                    
                    if biweek_str.startswith('BW-'):
                        try:
                            biweek_num = int(biweek_str.replace('BW-', ''))
                            
                            # Skip biweeks after target if in target year
                            if year == target_year and biweek_num > target_biweek:
                                continue
                            
                            # Load this biweek's data
                            response = s3_client.get_object(
                                Bucket=bucket_name,
                                Key=f'{SUBPRIME_INPUT_PREFIX}{year}/BW-{biweek_num}/data.parquet'
                            )
                            bw_data = pd.read_parquet(BytesIO(response['Body'].read()))
                            all_data.append(bw_data)
                            biweek_count += 1
                            
                            print(f"  ✓ Loaded {year}/BW-{biweek_num}: {len(bw_data)} rows")
                        except Exception as e:
                            print(f"  ⚠ Could not load {year}/{biweek_str}: {e}")
        
        if not all_data:
            raise Exception("No subprime data could be loaded from S3")
        
        # Step 4: Concatenate all data
        print(f"\n[DATA] Concatenating {biweek_count} biweekly dataset(s) + historical data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"  ✓ Combined dataset: {len(combined_data)} rows, {len(combined_data.columns)} columns")
        
        return combined_data
        
    except Exception as e:
        raise Exception(f"Failed to load subprime data: {e}")


def load_families_mapping(s3_client, bucket_name):
    """Load families mapping from S3."""
    try:
        print(f"\n[S3] Loading families mapping...")
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key=FAMILIES_MAPPING_KEY,
            )
            families = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded families mapping for {len(families)} families")
        except Exception as e:
            print(f"  ✗ Could not load families_mapping.json: {e}")
            raise
        
        return families
    except Exception as e:
        raise Exception(f"Failed to load jsons from S3: {e}")


def ensure_directories_exist():
    """Create temporary output directories."""
    print(f"\n[SETUP] Creating temporary directories")

    for directory in [FAMILY_DIR, STORE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def train_models(ts_per_family, ts_per_store):
    """Train SARIMAX models for each family and store."""
    
    smx_per_family = {}
    smx_per_store = {}

    smx_params = {
        'exog_features': SIGNIFICANT_EXOG,
        'order': (0, 1, 1),
        'seasonal_order': (0, 1, 1, 7),
        'trend': 'c'
    }

    start_time = dt.now(datetime.UTC)
    print(f"\n[TRAINING] Training SARIMAX models per family")
    for i, f in enumerate(ts_per_family, 1):
        print(f"  [{i}/{len(ts_per_family)}] {f}...", end=" ", flush=True)
        try:
            exog_data = ts_per_family[f][smx_params['exog_features']].drop(
                [col for col in smx_params['exog_features'] if len(ts_per_family[f][col].unique()) == 1],
                axis=1
            )
            
            model = SARIMAX(
                ts_per_family[f]['sales'],
                exog=exog_data,
                order=smx_params['order'],
                seasonal_order=smx_params['seasonal_order'],
                trend=smx_params['trend'],
            )
            smx_per_family[f] = model.fit(disp=False)
            print("✓")
            
        except Exception as e:
            print(f"✗")
            raise Exception(f"Failed to train family model for '{f}': {e}")
    
    print(f"  ✓ Trained {len(smx_per_family)} family models")

    print(f"\n[TRAINING] Training SARIMAX models per store")
    for i, s in enumerate(sorted(ts_per_store.keys()), 1):
        print(f"  [{i}/{len(ts_per_store)}] Store {s}...", end=" ", flush=True)
        try:
            exog_data = ts_per_store[s][smx_params['exog_features']].drop(
                [col for col in smx_params['exog_features'] if len(ts_per_store[s][col].unique()) == 1],
                axis=1
            )
            
            model = SARIMAX(
                ts_per_store[s]['sales'],
                exog=exog_data,
                order=smx_params['order'],
                seasonal_order=smx_params['seasonal_order'],
                trend=smx_params['trend'],
            )
            smx_per_store[s] = model.fit(disp=False)
            print("✓")
            
        except Exception as e:
            print(f"✗")
            raise Exception(f"Failed to train store model for store {s}: {e}")
    
    print(f"  ✓ Trained {len(smx_per_store)} store models")
    end_time = dt.now(datetime.UTC)
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n[TRAINING] Completed in {elapsed:.2f} seconds")

    return smx_per_family, smx_per_store, smx_params, start_time, end_time


def save_models(smx_per_family, smx_per_store):
    """Serialize and save models to temporary directories."""
    print(f"\n[SAVE] Serializing models to temporary directories")
    
    # Save family models
    print(f"  Saving {len(smx_per_family)} family models...")
    for f, model in smx_per_family.items():
        try:
            model_path = FAMILY_DIR / f"{family_encode(f)}.pkl"
            with open(model_path, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise Exception(f"Failed to save family model '{f}': {e}")
    print(f"    ✓ {len(smx_per_family)} family models saved")
    
    # Save store models
    print(f"  Saving {len(smx_per_store)} store models...")
    for s, model in smx_per_store.items():
        try:
            model_path = STORE_DIR / f"store_{s:02d}.pkl"
            with open(model_path, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise Exception(f"Failed to save store model for store {s}: {e}")
    print(f"    ✓ {len(smx_per_store)} store models saved")


def upload_to_s3(s3_client, model_bucket, year, biweek_num):
    """Upload trained models from temp directory to S3."""
    s3_prefix = f"{SARIMAX_OUTPUT_PREFIX}{year}/BW-{biweek_num}/"
    
    print(f"\n[S3] Uploading models to s3://{model_bucket}/{s3_prefix}")
    
    try:
        # Upload family models
        print(f"  Uploading family models...")
        for model_path in FAMILY_DIR.glob('*.pkl'):
            s3_key = f"{s3_prefix}family/{model_path.name}"

            with open(model_path, 'rb') as f_obj:
                s3_client.upload_fileobj(f_obj, model_bucket, s3_key)
        print(f"    ✓ Family models uploaded")
        
        # Upload store models
        print(f"  Uploading store models...")
        for model_path in STORE_DIR.glob('*.pkl'):
            s3_key = f"{s3_prefix}store/{model_path.name}"
            
            with open(model_path, 'rb') as f_obj:
                s3_client.upload_fileobj(f_obj, model_bucket, s3_key)
        print(f"    ✓ Store models uploaded")
        
        print(f"\n✓ Successfully uploaded models to S3")
        return True
        
    except ClientError as e:
        raise Exception(f"Error uploading to S3: {e}")
    

def upload_prime_data_to_s3(s3_client, bucket_name, prime_data, lambdas, hmvs, year, biweek_num):
    """Upload prime data and metadata to S3."""
    s3_prefix = f"{PRIME_OUTPUT_PREFIX}{year}/BW-{biweek_num}/"
    
    print(f"\n[S3] Uploading prime data to s3://{bucket_name}/{s3_prefix}")

    def save_json_to_s3(data, s3_key):
        buffer = BytesIO()
        json_data = json.dumps(data, indent=2)
        buffer.write(json_data.encode('utf-8'))
        buffer.seek(0)
        
        print(f"  Uploading {s3_key.split('/')[-1]}...", end=" ")
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        print(f"✓")
    
    try:
        # Upload parquet file
        parquet_buffer = BytesIO()
        prime_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_key = f"{s3_prefix}data.parquet"
        print(f"  Uploading data.parquet...", end=" ")
        s3_client.upload_fileobj(parquet_buffer, bucket_name, s3_key)
        print(f"✓")
        
        # Upload lambda values as JSON
        save_json_to_s3(lambdas, f"{s3_prefix}lambdas.json")

        # Upload HMV values as JSON
        save_json_to_s3(hmvs, f"{s3_prefix}hmvs.json")
        
        print(f"\n✓ Successfully uploaded prime data to S3")
        return True
        
    except ClientError as e:
        raise Exception(f"Error uploading to S3: {e}")


def log_job_metadata(dynamodb_resource, job_table_name, params, start_time, end_time, year, biweek_num):
    """Log job metadata to DynamoDB."""

    table = dynamodb_resource.Table(job_table_name)
    params = {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
    item = {
        'job_type': 'sarimax_training',
        'complete_timestamp': str(end_time)[:-6],
        'job_id': str(uuid.uuid4()),
        'elapsed_seconds': Decimal(str(f'{(end_time - start_time).total_seconds():.2f}')),
        'biweek': f'{year}-BW-{biweek_num}',
        'parameters': params,
    }
    
    try:
        table.put_item(Item=item)
        print(f"\n✓ Logged job metadata to DynamoDB")
    except ClientError as e:
        print(f"✗ Error logging to DynamoDB: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def cleanup_temp_files():
    """Remove temporary directory."""
    if TEMP_DIR.exists():
        print(f"\n[CLEANUP] Removing temporary directory: {TEMP_DIR}")
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✓ Cleaned up temporary files")
        except Exception as e:
            print(f"✗ Error removing {TEMP_DIR}: {e}")
            return False
    return True


if __name__ == '__main__':
    """
    ECS Fargate entry point for biweekly SARIMAX model training.
    
    Args:
        env_name: AWS CDK environment name (e.g., 'dev', 'prod')
    """
    env_name = os.environ.get('ENVIRONMENT', 'dev')

    # Get S3 bucket names and configuration from environment variables
    data_bucket_name = os.environ.get('DATA_BUCKET')
    model_bucket_name = os.environ.get('MODEL_BUCKET')
    job_table_name = os.environ.get('JOB_TABLE')
    year_str = os.environ.get('YEAR')
    biweek_num_str = os.environ.get('BIWEEK_NUM')

    if not data_bucket_name:
        print("Error: DATA_BUCKET environment variable not set")
        sys.exit(1)
    if not model_bucket_name:
        print("Error: MODEL_BUCKET environment variable not set")
        sys.exit(1)
    if not job_table_name:
        print("Error: JOB_TABLE environment variable not set")
        sys.exit(1)
    if not year_str:
        print("Error: YEAR environment variable not set")
        sys.exit(1)
    if not biweek_num_str:
        print("Error: BIWEEK_NUM environment variable not set")
        sys.exit(1)
    
    # Convert to integers
    try:
        year = int(year_str)
        biweek_num = int(biweek_num_str)
    except ValueError:
        print(f"Error: YEAR and BIWEEK_NUM must be integers. Got YEAR={year_str}, BIWEEK_NUM={biweek_num_str}")
        sys.exit(1)
    
    print("=" * 70)
    print("Biweekly SARIMAX Model Training")
    print("=" * 70)
    print(f"Environment: {env_name}")
    print(f"Data Bucket: {data_bucket_name}")
    print(f"Model Bucket: {model_bucket_name}")
    print(f"Temp Directory: {TEMP_DIR}")
    print()
    
    try:
        # Setup
        ensure_directories_exist()
        
        # Initialize boto3 clients
        s3_client = boto3.client('s3')
        dynamodb_resource = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
        
        # Step 1: Load biweekly and historical subprime data up to target biweek
        print("[1/8] Loading biweekly and historical subprime data...")
        subprime_data = load_subprime_data(s3_client, data_bucket_name, year, biweek_num)
        print(f"✓ Loaded data up to biweek: {year}/BW-{biweek_num}")
        
        # Step 2: Check if model training has already been completed
        print("\n[2/8] Checking if model training has already been completed...")
        s3_output_prefix = f"{SARIMAX_OUTPUT_PREFIX}{year}/BW-{biweek_num}/"
        if marker_exists(s3_client, model_bucket_name, s3_output_prefix):
            print("✗ Error: Model training workflow has already been completed.")
            print(f"  Models exist at s3://{model_bucket_name}/{s3_output_prefix}{MARKER}")
            sys.exit(0)
        print("✓ No marker found. Ready to proceed with training.")
        
        # Step 3: Load families mapping
        print("\n[3/8] Loading families mapping...")
        families = load_families_mapping(s3_client, data_bucket_name)
        
        # Step 4: Apply prime transformations
        print("\n[4/8] Applying prime transformations...")
        prime_data, lambdas, hmvs = fit_prime_transform(subprime_data)
        
        # Step 5: Upload prime data to S3
        print("\n[5/8] Uploading prime data to S3...")
        upload_prime_data_to_s3(s3_client, data_bucket_name, prime_data, lambdas, hmvs, year, biweek_num)
        
        # Step 6: Build time series and train models
        print("\n[6/8] Building time series and training SARIMAX models...")
        ts_per_family, ts_per_store = build_time_series(prime_data)
        smx_per_family, smx_per_store, params, start_time, end_time = train_models(ts_per_family, ts_per_store)
        
        # Step 7: Save and upload models
        print("\n[7/8] Saving and uploading models to S3...")
        save_models(smx_per_family, smx_per_store)
        upload_to_s3(s3_client, model_bucket_name, year, biweek_num)
        
        # Step 8: Log job metadata and write marker
        print("\n[8/8] Logging job metadata...")
        log_job_metadata(dynamodb_resource, job_table_name, params, start_time, end_time, year, biweek_num)
        
        # Write completion markers
        write_marker(s3_client, data_bucket_name, f"{PRIME_OUTPUT_PREFIX}{year}/BW-{biweek_num}/")
        write_marker(s3_client, model_bucket_name, s3_output_prefix)
        
        # Cleanup
        cleanup_temp_files()
        
        print("\n" + "=" * 70)
        print("✓ Biweekly SARIMAX Model Training Completed Successfully!")
        print("=" * 70)
        print(f"Prime data uploaded to: s3://{data_bucket_name}/{PRIME_OUTPUT_PREFIX}{year}/BW-{biweek_num}/")
        print(f"Models uploaded to: s3://{model_bucket_name}/{s3_output_prefix}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        cleanup_temp_files()
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        cleanup_temp_files()
        sys.exit(1)
