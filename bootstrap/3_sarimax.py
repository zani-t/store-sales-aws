#!/usr/bin/env python3
"""
Train SARIMAX models per family and per store on processed historical data.
Downloads processed data from S3, trains models, saves them to temp directory,
and uploads to S3. Models are stored in <model_bucket>/sarimax/historical/.
"""

import os
import sys
import pickle
import shutil
import warnings
import tempfile
from pathlib import Path
from io import BytesIO

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

from bootstrap import get_stack_output, marker_exists, write_marker

# Suppress warnings
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Temporary directory for local model storage
TEMP_DIR = Path(tempfile.mkdtemp(prefix='sarimax_models_'))
FAMILY_DIR = TEMP_DIR / 'family'
STORE_DIR = TEMP_DIR / 'store'

# S3 output paths
S3_OUTPUT_PREFIX = 'sarimax/historical/'

# Exogenous variables used in model training
SIGNIFICANT_EXOG = ['hmv', 'exists_promotion', 'exists_transaction']


def ensure_directories_exist():
    """Create temporary output directories."""
    print(f"\n[SETUP] Creating temporary directories")
    
    for directory in [FAMILY_DIR, STORE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def download_from_s3(s3_client, bucket_name):
    """Download processed parquet file from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
    
    Returns:
        pd.DataFrame: Processed training data
    """
    print(f"\n[S3] Downloading processed data from s3://{bucket_name}/processed/sarimax-prime/historical/")
    
    s3_key = "processed/sarimax-prime/historical/data.parquet"
    
    try:
        print(f"  Downloading {s3_key}...", end=" ")
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        # Read parquet from S3 directly into DataFrame
        train = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"✓")
        print(f"  Loaded {len(train)} rows, {len(train.columns)} columns")
        
        return train
        
    except ClientError as e:
        print(f"✗")
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"  Error: File not found at s3://{bucket_name}/{s3_key}")
            print(f"  Please ensure 2_sarimax_prime.py has been run first.")
        else:
            print(f"  Error: {e}")
        return None
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return None


def build_time_series(train):
    """Build time series aggregations per family and per store.
    
    Args:
        train: DataFrame with processed training data
    
    Returns:
        tuple: (ts_per_family dict, ts_per_store dict)
    """
    print(f"\n[TIMESERIES] Building time series aggregations")
    
    exog = {feature: 'mean' for feature in ['sales'] + SIGNIFICANT_EXOG}
    
    ts_per_family = {}
    ts_per_store = {}
    
    print(f"  Building time series per family...")
    for f in train['family'].unique():
        ts_per_family[f] = train.loc[train['family'] == f].groupby(['date']).agg(exog)
    print(f"    ✓ {len(ts_per_family)} families")
    
    print(f"  Building time series per store...")
    for s in range(1, 55):
        store_data = train.loc[train['store_nbr'] == s]
        if len(store_data) > 0:
            ts_per_store[s] = store_data.groupby(['date']).agg(exog)
    print(f"    ✓ {len(ts_per_store)} stores")
    
    return ts_per_family, ts_per_store


def train_family_models(ts_per_family):
    """Train SARIMAX models for each family. Abort on any failure.
    
    Args:
        ts_per_family: Dictionary of time series per family
    
    Returns:
        dict: Trained models per family
    
    Raises:
        Exception: If any model fails to train
    """
    print(f"\n[TRAINING] Training SARIMAX models per family")
    
    smx_per_family = {}
    
    for i, f in enumerate(ts_per_family, 1):
        print(f"  [{i}/{len(ts_per_family)}] {f}...", end=" ", flush=True)
        
        try:
            exog_data = ts_per_family[f][SIGNIFICANT_EXOG].drop(
                [col for col in SIGNIFICANT_EXOG if len(ts_per_family[f][col].unique()) == 1],
                axis=1
            )
            
            model = SARIMAX(
                ts_per_family[f]['sales'],
                exog=exog_data,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 7),
                trend='c',
            )
            smx_per_family[f] = model.fit(disp=False)
            print("✓")
            
        except Exception as e:
            print(f"✗")
            raise Exception(f"Failed to train family model for '{f}': {e}")
    
    print(f"  ✓ Trained {len(smx_per_family)} family models")
    return smx_per_family



def train_store_models(ts_per_store):
    """Train SARIMAX models for each store. Abort on any failure.
    
    Args:
        ts_per_store: Dictionary of time series per store
    
    Returns:
        dict: Trained models per store
    
    Raises:
        Exception: If any model fails to train
    """
    print(f"\n[TRAINING] Training SARIMAX models per store")
    
    smx_per_store = {}
    
    for i, s in enumerate(sorted(ts_per_store.keys()), 1):
        print(f"  [{i}/{len(ts_per_store)}] Store {s}...", end=" ", flush=True)
        
        try:
            exog_data = ts_per_store[s][SIGNIFICANT_EXOG].drop(
                [col for col in SIGNIFICANT_EXOG if len(ts_per_store[s][col].unique()) == 1],
                axis=1
            )
            
            model = SARIMAX(
                ts_per_store[s]['sales'],
                exog=exog_data,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 7),
                trend='c',
            )
            smx_per_store[s] = model.fit(disp=False)
            print("✓")
            
        except Exception as e:
            print(f"✗")
            raise Exception(f"Failed to train store model for store {s}: {e}")
    
    print(f"  ✓ Trained {len(smx_per_store)} store models")
    return smx_per_store


def save_models(smx_per_family, smx_per_store):
    """Serialize and save models to temporary directories.
    
    Args:
        smx_per_family: Dictionary of trained family models
        smx_per_store: Dictionary of trained store models
    """
    print(f"\n[SAVE] Serializing models to temporary directories")
    
    # Save family models
    print(f"  Saving {len(smx_per_family)} family models...")
    for f, model in smx_per_family.items():
        try:
            model_path = FAMILY_DIR / f"{f.replace('/', '_').replace(' ', '_')}.pkl"
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


def upload_to_s3(s3_client, model_bucket):
    """Upload trained models from temp directory to S3.
    
    Args:
        s3_client: Boto3 S3 client
        model_bucket: S3 model bucket name
        smx_per_family: Dictionary of trained family models
        smx_per_store: Dictionary of trained store models
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[S3] Uploading models to s3://{model_bucket}/{S3_OUTPUT_PREFIX}")
    
    try:
        # Upload family models
        print(f"  Uploading family models...")
        for model_path in FAMILY_DIR.glob('*.pkl'):
            s3_key = f"{S3_OUTPUT_PREFIX}family/{model_path.name}"

            with open(model_path, 'rb') as f_obj:
                s3_client.upload_fileobj(f_obj, model_bucket, s3_key)
        print(f"    ✓ Family models uploaded")
        
        # Upload store models
        print(f"  Uploading store models...")
        for model_path in STORE_DIR.glob('*.pkl'):
            s3_key = f"{S3_OUTPUT_PREFIX}store/{model_path.name}"
            
            with open(model_path, 'rb') as f_obj:
                s3_client.upload_fileobj(f_obj, model_bucket, s3_key)
        print(f"    ✓ Store models uploaded")
        
        print(f"\n✓ Successfully uploaded models to S3")
        return True
        
    except ClientError as e:
        print(f"✗")
        print(f"  Error uploading to S3: {e}")
        return False
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return False


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



def main(env_name):
    """Main entry point.
    
    Args:
        env_name: AWS CDK environment name (e.g., 'dev', 'prod')
    """
    # Get S3 bucket names from CDK CloudFormation exports
    try:
        data_bucket_name = get_stack_output(env_name, f"{env_name}-DataBucketName")
        model_bucket_name = get_stack_output(env_name, f"{env_name}-ModelBucketName")
    except Exception as e:
        print(f"Error retrieving S3 bucket names: {e}")
        print(f"Please ensure CDK stack '{env_name}-StorageStack' exists.")
        sys.exit(1)
    
    print("=" * 60)
    print("SARIMAX Model Training")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Data Bucket: {data_bucket_name}")
    print(f"Model Bucket: {model_bucket_name}")
    print(f"Temp Directory: {TEMP_DIR}")
    print()
    
    try:
        # Setup
        ensure_directories_exist()
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Check for marker file indicating processed data is ready
        print("\n[1/7] Checking for processed data marker...")
        if not marker_exists(data_bucket_name, "processed/sarimax-prime/historical/"):
            print("✗ Error: Processed data not found at s3://{}/processed/sarimax-prime/historical/_COMPLETE".format(data_bucket_name))
            print("  Please run: python bootstrap/2_sarimax_prime.py {}".format(env_name))
            sys.exit(1)
        print("✓ Marker found. Processed data is ready.")
        
        # Check if model training has already been completed
        print("\n[2/7] Checking if model training has already been completed...")
        if marker_exists(model_bucket_name, S3_OUTPUT_PREFIX):
            print("✗ Error: Model training workflow has already been completed.")
            print(f"  Models exist at s3://{model_bucket_name}/{S3_OUTPUT_PREFIX}_COMPLETE")
            sys.exit(0)
        print("✓ No marker found. Ready to proceed with training.")
        
        # Download from S3
        print("\n[3/7] Downloading processed data from S3...")
        train = download_from_s3(s3_client, data_bucket_name)
        
        if train is None:
            print("Error: Failed to download data from S3")
            sys.exit(1)
        
        # Build time series
        print("\n[4/7] Building time series aggregations...")
        ts_per_family, ts_per_store = build_time_series(train)
        
        # Train models
        print("\n[5/7] Training SARIMAX models...")
        smx_per_family = train_family_models(ts_per_family)
        smx_per_store = train_store_models(ts_per_store)
        
        # Save models to temp directory
        print("\n[6/7] Saving models to temporary directory...")
        save_models(smx_per_family, smx_per_store)
        
        # Upload to S3
        print("\n[7/7] Uploading models to S3...")
        upload_success = upload_to_s3(s3_client, model_bucket_name)
        
        if not upload_success:
            print("Error: Failed to upload models to S3")
            sys.exit(1)
        
        # Write marker and cleanup
        write_marker(model_bucket_name, S3_OUTPUT_PREFIX)
        cleanup_temp_files()
        
        print("\n" + "=" * 60)
        print("✓ Model training completed successfully!")
        print("=" * 60)
        print(f"Models uploaded to: s3://{model_bucket_name}/{S3_OUTPUT_PREFIX}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        cleanup_temp_files()
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        cleanup_temp_files()
        sys.exit(1)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 3_sarimax.py <env_name>")
        print("Example: python 3_sarimax.py dev")
        sys.exit(1)
    
    env_name = sys.argv[1]
    main(env_name)

