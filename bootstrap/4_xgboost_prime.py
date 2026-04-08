#!/usr/bin/env python3
"""
Prepare XGBoost training data by loading SARIMAX model predictions.
Downloads processed data from S3, loads trained SARIMAX models, generates
predictions, and creates a feature-rich dataset for XGBoost modeling.
Output data is stored in <data_bucket>/processed/xgboost-prime/historical/.
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from io import BytesIO

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError

from bootstrap import (
    get_stack_output,
    marker_exists,
    write_marker,
    load_time_series,
    SIGNIFICANT_EXOG
    )

# Suppress warnings
warnings.filterwarnings('ignore')

# S3 paths
SARIMAX_MODEL_PREFIX = 'sarimax/historical/'
S3_OUTPUT_PREFIX = 'processed/xgboost-prime/historical/'


def download_from_s3(s3_client, bucket_name, s3_key):
    """Download file from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        s3_key: S3 object key
    
    Returns:
        BytesIO object or None if error
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return BytesIO(response['Body'].read())
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"    ✗ File not found: s3://{bucket_name}/{s3_key}")
        else:
            print(f"    ✗ Error: {e}")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def download_processed_data(s3_client, data_bucket):
    """Download processed parquet file from S3.
    
    Args:
        s3_client: Boto3 S3 client
        data_bucket: S3 data bucket name
    
    Returns:
        pd.DataFrame: Processed training data
    """
    print(f"\n[S3] Downloading processed data from s3://{data_bucket}/{SARIMAX_MODEL_PREFIX.replace('sarimax', 'processed/sarimax-prime')}")
    
    s3_key = "processed/sarimax-prime/historical/data.parquet"
    
    try:
        print(f"  Downloading {s3_key}...", end=" ", flush=True)
        data_bytes = download_from_s3(s3_client, data_bucket, s3_key)
        
        if data_bytes is None:
            return None
        
        train = pd.read_parquet(data_bytes)
        print(f"✓")
        print(f"  Loaded {len(train)} rows, {len(train.columns)} columns")
        
        return train
        
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return None


def download_sarimax_models(s3_client, model_bucket):
    """Download SARIMAX models from S3.
    
    Args:
        s3_client: Boto3 S3 client
        model_bucket: S3 model bucket name
    
    Returns:
        tuple: (family_models dict, store_models dict)
    """
    print(f"\n[S3] Downloading SARIMAX models from s3://{model_bucket}/{SARIMAX_MODEL_PREFIX}")
    
    smx_per_family = {}
    smx_per_store = {}
    
    try:
        # List and download family models
        print(f"  Downloading family models...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=model_bucket, Prefix=f"{SARIMAX_MODEL_PREFIX}family/")
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.pkl'):
                    model_name = Path(obj['Key']).stem
                    data_bytes = download_from_s3(s3_client, model_bucket, obj['Key'])
                    if data_bytes:
                        smx_per_family[model_name] = pickle.load(data_bytes)
        
        print(f"    ✓ Downloaded {len(smx_per_family)} family models")
        
        # List and download store models
        print(f"  Downloading store models...")
        pages = paginator.paginate(Bucket=model_bucket, Prefix=f"{SARIMAX_MODEL_PREFIX}store/")
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.pkl'):
                    # Extract store number from filename (e.g., "store_01.pkl" -> 1)
                    model_name = Path(obj['Key']).stem
                    if model_name.startswith('store_'):
                        store_num = int(model_name.split('_')[1])
                        data_bytes = download_from_s3(s3_client, model_bucket, obj['Key'])
                        if data_bytes:
                            smx_per_store[store_num] = pickle.load(data_bytes)
        
        print(f"    ✓ Downloaded {len(smx_per_store)} store models")
        
        return smx_per_family, smx_per_store
        
    except Exception as e:
        print(f"  ✗ Error downloading models: {e}")
        return None, None


def generate_inferences(ts_per_family, ts_per_store, smx_per_family, smx_per_store):
    """Generate predictions from loaded SARIMAX models using time series data.
    
    Args:
        ts_per_family: Dictionary of aggregated family time series
        ts_per_store: Dictionary of aggregated store time series
        smx_per_family: Dictionary of trained family models
        smx_per_store: Dictionary of trained store models
    
    Returns:
        tuple: (family_inferences dict, store_inferences dict)
    
    Raises:
        Exception: If inference generation fails
    """
    print(f"\n[INFERENCE] Generating SARIMAX predictions")
    
    inf_per_family = {}
    inf_per_store = {}
    
    # Generate family predictions
    print(f"  Generating predictions per family...")
    for f in smx_per_family:
        try:
            if f not in ts_per_family or len(ts_per_family[f]) == 0:
                continue
            
            exog_data = ts_per_family[f][SIGNIFICANT_EXOG].drop(
                [col for col in SIGNIFICANT_EXOG if len(ts_per_family[f][col].unique()) == 1],
                axis=1
            )
            inference = smx_per_family[f].get_prediction(
                start=exog_data.index[0],
                end=exog_data.index[-1],
                exog=exog_data
            )
            inf_per_family[f] = pd.DataFrame(inference.predicted_mean)
        except Exception as e:
            raise Exception(f"Failed to generate inference for family '{f}': {e}")
    
    print(f"    ✓ Generated predictions for {len(inf_per_family)} families")
    
    # Generate store predictions
    print(f"  Generating predictions per store...")
    for s in smx_per_store:
        try:
            if s not in ts_per_store or len(ts_per_store[s]) == 0:
                continue
            
            exog_data = ts_per_store[s][SIGNIFICANT_EXOG].drop(
                [col for col in SIGNIFICANT_EXOG if len(ts_per_store[s][col].unique()) == 1],
                axis=1
            )
            inference = smx_per_store[s].get_prediction(
                start=exog_data.index[0],
                end=exog_data.index[-1],
                exog=exog_data
            )
            inf_per_store[s] = pd.DataFrame(inference.predicted_mean)
        except Exception as e:
            raise Exception(f"Failed to generate inference for store {s}: {e}")
    
    print(f"    ✓ Generated predictions for {len(inf_per_store)} stores")
    
    return inf_per_family, inf_per_store


def create_feature_dataframe(train, inf_per_family, inf_per_store):
    """Create complete feature dataframe with SARIMAX predictions.
    
    Args:
        train: Original training DataFrame
        inf_per_family: Dictionary of family predictions
        inf_per_store: Dictionary of store predictions
    
    Returns:
        pd.DataFrame: Feature-rich DataFrame
    """
    print(f"\n[FEATURES] Creating feature-rich dataframe")
    
    try:
        X = train.copy()
        
        # Merge family time series predictions
        print(f"  Merging family time series predictions...")
        ts_family_df = pd.concat([
            ts_df.reset_index().assign(family=fam)
            for fam, ts_df in inf_per_family.items()
        ], ignore_index=True)
        
        if 'index' in ts_family_df.columns:
            ts_family_df.drop('index', axis=1, inplace=True)
        ts_family_df = ts_family_df.rename(columns={'predicted_mean': 'ts_family'})
        
        X = X.merge(ts_family_df, on=['date', 'family'], how='left')
        X['ts_family_active'] = X['ts_family'].notna().astype('int8')
        X['ts_family'] = X['ts_family'].fillna(0)
        
        # Merge store time series predictions
        print(f"  Merging store time series predictions...")
        ts_store_df = pd.concat([
            ts_df.reset_index().assign(store_nbr=store)
            for store, ts_df in inf_per_store.items()
        ], ignore_index=True)
        
        if 'index' in ts_store_df.columns:
            ts_store_df.drop('index', axis=1, inplace=True)
        ts_store_df = ts_store_df.rename(columns={'predicted_mean': 'ts_store'})
        
        X = X.merge(ts_store_df, on=['date', 'store_nbr'], how='left')
        X['ts_store_active'] = X['ts_store'].notna().astype('int8')
        X['ts_store'] = X['ts_store'].fillna(0)
        
        # Feature engineering
        print(f"  Applying feature engineering...")
        X['store_nbr'] = X['store_nbr'].apply(str)
        X['cluster'] = X['cluster'].apply(str)
        X['month'] = X['date'].dt.month
        X['day_of_month'] = X['date'].dt.day
        X['day_of_week'] = X['date'].dt.day_of_week
        
        X = X.drop('date', axis=1)
        X = X.drop('transactions', axis=1)
        
        X = pd.get_dummies(X, columns=[
            'store_nbr', 'cluster', 'family', 'city', 'state',
            'month', 'day_of_month', 'day_of_week'
        ])
        
        print(f"    ✓ Created dataframe with {len(X)} rows, {len(X.columns)} columns")
        
        return X
        
    except Exception as e:
        print(f"  ✗ Error creating feature dataframe: {e}")
        return None


def upload_to_s3(s3_client, data_bucket, X):
    """Upload processed dataframe to S3.
    
    Args:
        s3_client: Boto3 S3 client
        data_bucket: S3 data bucket name
        X: DataFrame to upload
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[S3] Uploading processed data to s3://{data_bucket}/{S3_OUTPUT_PREFIX}")
    
    try:
        s3_key = f"{S3_OUTPUT_PREFIX}data.parquet"
        
        # Convert dataframe to parquet bytes
        parquet_bytes = BytesIO()
        X.to_parquet(parquet_bytes, index=False)
        parquet_bytes.seek(0)
        
        print(f"  Uploading {s3_key}...", end=" ", flush=True)
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=parquet_bytes.getvalue()
        )
        print(f"✓")
        
        print(f"\n✓ Successfully uploaded data to S3")
        return True
        
    except ClientError as e:
        print(f"✗")
        print(f"  Error uploading to S3: {e}")
        return False
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return False


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
    print("XGBoost Prime Data Preparation")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Data Bucket: {data_bucket_name}")
    print(f"Model Bucket: {model_bucket_name}")
    print()
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Check for marker file indicating processed data is ready
        print("\n[1/9] Checking for processed data marker...")
        if not marker_exists(data_bucket_name, "processed/sarimax-prime/historical/"):
            print("✗ Processed data marker not found.")
            print("  Please ensure 2_sarimax_prime.py and 3_sarimax.py have been run first.")
            sys.exit(1)
        print("✓ Marker found. Processed data is ready.")
        
        # Check for SARIMAX model completion marker
        print("\n[2/9] Checking for SARIMAX model completion marker...")
        if not marker_exists(model_bucket_name, SARIMAX_MODEL_PREFIX):
            print("✗ SARIMAX model completion marker not found.")
            print("  Please ensure 3_sarimax.py has been run successfully.")
            sys.exit(1)
        print("✓ Marker found. SARIMAX models are ready.")
        
        # Check if XGBoost data preparation has already been completed
        print("\n[3/9] Checking if XGBoost data preparation has already been completed...")
        if marker_exists(data_bucket_name, S3_OUTPUT_PREFIX):
            print("✓ Marker found. Data preparation already completed.")
            print(f"  Output: s3://{data_bucket_name}/{S3_OUTPUT_PREFIX}")
            return
        print("✓ No marker found. Ready to proceed with data preparation.")
        
        # Download processed data from S3
        print("\n[4/9] Downloading processed data from S3...")
        train = download_processed_data(s3_client, data_bucket_name)
        
        if train is None:
            print("✗ Failed to download processed data.")
            sys.exit(1)
        
        # Download SARIMAX models from S3
        print("\n[5/9] Downloading SARIMAX models from S3...")
        smx_per_family, smx_per_store = download_sarimax_models(s3_client, model_bucket_name)
        
        if smx_per_family is None or smx_per_store is None:
            print("✗ Failed to download SARIMAX models.")
            sys.exit(1)
        
        if len(smx_per_family) == 0 and len(smx_per_store) == 0:
            print("✗ No SARIMAX models found in S3.")
            sys.exit(1)
        
        # Load time series per family and store
        print("\n[6/9] Loading time series from S3...")
        ts_per_family, ts_per_store = load_time_series(s3_client, data_bucket_name)
        
        if ts_per_family is None or ts_per_store is None:
            print("✗ Failed to load time series.")
            sys.exit(1)
        
        # Generate inferences
        print("\n[7/9] Generating SARIMAX predictions...")
        try:
            inf_per_family, inf_per_store = generate_inferences(
                ts_per_family, ts_per_store, smx_per_family, smx_per_store
            )
        except Exception as e:
            print(f"✗ Failed to generate inferences: {e}")
            sys.exit(1)
        
        # Create feature dataframe
        print("\n[8/9] Creating feature-rich dataframe...")
        X = create_feature_dataframe(train, inf_per_family, inf_per_store)
        
        if X is None:
            print("✗ Failed to create feature dataframe.")
            sys.exit(1)
        
        # Upload to S3
        print("\n[9/9] Uploading processed data to S3...")
        upload_success = upload_to_s3(s3_client, data_bucket_name, X)
        
        if not upload_success:
            print("✗ Failed to upload data to S3.")
            sys.exit(1)
        
        # Write marker and cleanup
        write_marker(data_bucket_name, S3_OUTPUT_PREFIX)
        
        print("\n" + "=" * 60)
        print("✓ XGBoost data preparation completed successfully!")
        print("=" * 60)
        print(f"Output: s3://{data_bucket_name}/{S3_OUTPUT_PREFIX}data.parquet")
        print()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 4_xgboost_prime.py <env_name>")
        print("Example: python 4_xgboost_prime.py dev")
        sys.exit(1)
    
    env_name = sys.argv[1]
    main(env_name)

