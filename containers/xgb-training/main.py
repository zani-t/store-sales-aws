#!/usr/bin/env python3
"""
Train XGBoost-based StackingRegressor model on biweekly data.
Loads SARIMAX prime data from S3, generates SARIMAX predictions,
creates feature-rich dataset, trains stacking ensemble model,
and saves to S3. Output data is stored in <data_bucket>/processed/xgboost-prime/biweekly/<year>/BW-<biweek>/
and model in <model_bucket>/xgboost/biweekly/<year>/BW-<biweek>/.
"""

from io import BytesIO
from decimal import Decimal
from pathlib import Path

import datetime
import os
import sys
import warnings
import boto3
import pandas as pd
import numpy as np
import pickle
import joblib
import uuid
import gc
from datetime import datetime as dt
from botocore.exceptions import ClientError
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

from tsf2_core.constants import (
    MARKER,
    MODEL_FILENAME,
    PRIME_BIWEEKLY_PREFIX,
    SARIMAX_MODEL_BIWEEKLY_PREFIX,
    SIGNIFICANT_EXOG,
    XGBOOST_MODEL_BIWEEKLY_PREFIX,
    XGBOOST_PRIME_BIWEEKLY_PREFIX,
)
from tsf2_core.s3 import marker_exists, write_marker
from tsf2_core.timeseries import load_time_series

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SARIMAX_PRIME_PREFIX = PRIME_BIWEEKLY_PREFIX
SARIMAX_MODEL_PREFIX = SARIMAX_MODEL_BIWEEKLY_PREFIX
XGBOOST_PRIME_PREFIX = XGBOOST_PRIME_BIWEEKLY_PREFIX
XGBOOST_MODEL_PREFIX = XGBOOST_MODEL_BIWEEKLY_PREFIX


def load_sarimax_prime_data(s3_client, bucket_name, year, biweek_num):
    """Load latest biweekly SARIMAX prime data from S3.
    
    Returns:
        tuple: (data DataFrame, latest_year, latest_biweek)
    """
    try:
        # Load only the specified biweek's data
        print(f"  Loading data for {year}/BW-{biweek_num}...")
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=f'{SARIMAX_PRIME_PREFIX}{year}/BW-{biweek_num}/data.parquet'
        )
        data = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"  ✓ Loaded {len(data)} rows, {len(data.columns)} columns")
        
        return data
        
    except Exception as e:
        raise Exception(f"Failed to load SARIMAX prime data: {e}")


def download_sarimax_models(s3_client, model_bucket, year, biweek_num):
    """Download SARIMAX models from S3 for specified biweek.
    
    Returns:
        tuple: (family_models dict, store_models dict)
    """
    print(f"\n[S3] Downloading SARIMAX models from s3://{model_bucket}/{SARIMAX_MODEL_PREFIX}{year}/BW-{biweek_num}/")
    
    smx_per_family = {}
    smx_per_store = {}
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Download family models
        print(f"  Downloading family models...")
        pages = paginator.paginate(Bucket=model_bucket, Prefix=f"{SARIMAX_MODEL_PREFIX}{year}/BW-{biweek_num}/family/")
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.pkl'):
                    try:
                        response = s3_client.get_object(Bucket=model_bucket, Key=key)
                        smx_per_family[key.split('/')[-1].replace('.pkl', '')] = pickle.loads(response['Body'].read())
                    except Exception as e:
                        print(f"    ⚠ Could not load {key}: {e}")
        
        print(f"    ✓ Downloaded {len(smx_per_family)} family models")
        
        # Download store models
        print(f"  Downloading store models...")
        pages = paginator.paginate(Bucket=model_bucket, Prefix=f"{SARIMAX_MODEL_PREFIX}{year}/BW-{biweek_num}/store/")
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.pkl'):
                    try:
                        response = s3_client.get_object(Bucket=model_bucket, Key=key)
                        filename = key.split('/')[-1].replace('.pkl', '')
                        store_num = int(filename.replace('store_', ''))
                        smx_per_store[store_num] = pickle.loads(response['Body'].read())
                    except Exception as e:
                        print(f"    ⚠ Could not load {key}: {e}")
        
        print(f"    ✓ Downloaded {len(smx_per_store)} store models")
        
        return smx_per_family, smx_per_store
        
    except Exception as e:
        raise Exception(f"Failed to download SARIMAX models: {e}")


def generate_inferences(ts_per_family, ts_per_store, smx_per_family, smx_per_store):
    """Generate SARIMAX predictions to use as features.
    
    Returns:
        tuple: (family_predictions DataFrame, store_predictions DataFrame)
    """
    print(f"\n[INFERENCE] Generating SARIMAX predictions as features")
    
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


def create_feature_dataframe(X, inf_per_family, inf_per_store):
    """Create feature dataframe for XGBoost training using SARIMAX predictions.
    
    Returns:
        tuple: (X DataFrame, y Series)
    """
    print(f"\n[FEATURES] Creating XGBoost feature dataframe")
    
    try:
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

        if 'sales' in X.columns:
            y = X.pop('sales').to_frame()
            X_cols = X.columns.tolist()
            print(f"    ✓ Created dataframe with {len(X)} rows, {len(X.columns)} columns")
            return X, y, X_cols
        else:
            raise Exception("Target variable 'sales' not found in prime data")
        
    except Exception as e:
        raise Exception(f"Failed to create XGBoost features: {e}")


def train_stacking_model(X, y):
    """Train stacked XGBoost regression model.
    
    Returns:
        StackingRegressor: Trained model
    """
    xgbsr_params = {
        'estimators': {
            'xgb1': {
                'model': 'XGBRegressor',
                'params': {
                    'n_estimators': 100,
                    'learning_rate': Decimal('0.1'),
                    'seed': 66
                }
            },
            'xgb2': {
                'model': 'XGBRegressor',
                'params': {
                    'n_estimators': 100,
                    'learning_rate': Decimal('0.1'),
                    'seed': 77
                }
            }
       },
        'final_estimator': {
            'model': 'LinearRegression'
        }
    }

    print(f"\n[TRAINING] Training stacked XGBoost model")
    
    try:
        start_time = dt.now(datetime.UTC)
        print(f"  Input shape: X={X.shape}, y={y.shape}")
        
        # Define base estimators - two XGBoost models with different random seeds
        estimators = [
            ('xgb1', XGBRegressor(n_estimators=100, learning_rate=0.1, seed=66)),
            ('xgb2', XGBRegressor(n_estimators=100, learning_rate=0.1, seed=77)),
        ]
        
        # Create stacking regressor with LinearRegression as final estimator
        sr = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            verbose=0
        )
        
        print(f"  Fitting stacking ensemble...")
        sr.fit(X, y.values.ravel() if isinstance(y, pd.DataFrame) else y)
        print(f"  ✓ Model training completed")
        end_time = dt.now(datetime.UTC)
        elapsed = (end_time - start_time).total_seconds()
        print(f"  Training time: {elapsed:.2f} seconds")
        
        return sr, xgbsr_params, start_time, end_time
        
    except Exception as e:
        raise Exception(f"Failed to train stacking model: {e}")


def serialize_model(model, X_cols):
    """Serialize trained model to bytes using joblib.
    
    Args:
        model: Trained sklearn model
        X_cols: List of feature column names

    Returns:
        BytesIO: Serialized model or None if error
    """
    print(f"\n[SERIALIZATION] Serializing model")
    
    try:
        print(f"  Serializing using joblib...", end=" ", flush=True)
        model_bytes = BytesIO()
        joblib.dump({
            'model': model,
            'feature_names': X_cols
        }, model_bytes)
        model_bytes.seek(0)
        print(f"✓")
        print(f"  Serialized size: {len(model_bytes.getvalue()) / 1024 / 1024:.2f} MB")
        
        return model_bytes
        
    except Exception as e:
        print(f"✗")
        print(f"  Error serializing model: {e}")
        return None


def upload_xgboost_prime_data(s3_client, data_bucket, prime_data, year, biweek_num):
    """Upload XGBoost-prime feature data to S3."""
    s3_prefix = f"{XGBOOST_PRIME_PREFIX}{year}/BW-{biweek_num}/"
    
    print(f"\n[S3] Uploading XGBoost-prime data to s3://{data_bucket}/{s3_prefix}")
    
    try:
        s3_key = f"{s3_prefix}data.parquet"
        
        parquet_bytes = BytesIO()
        prime_data.to_parquet(parquet_bytes, index=False)
        parquet_bytes.seek(0)
        
        print(f"  Uploading {s3_key}...", end=" ", flush=True)
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=parquet_bytes.getvalue()
        )
        print(f"✓")
        print(f"    Size: {len(prime_data) * 100 / 1024 / 1024:.2f} MB (estimated)")
        
    except Exception as e:
        raise Exception(f"Failed to upload XGBoost-prime data: {e}")


def upload_model_to_s3(s3_client, model_bucket, model_bytes, year, biweek_num):
    """Upload trained model to S3.
    
    Returns:
        str: S3 path of uploaded model
    """
    s3_prefix = f"{XGBOOST_MODEL_PREFIX}{year}/BW-{biweek_num}/"
    
    print(f"\n[S3] Uploading model to s3://{model_bucket}/{s3_prefix}")
    
    try:
        s3_key = f"{s3_prefix}{MODEL_FILENAME}"
        
        print(f"  Uploading {s3_key}...", end=" ", flush=True)
        s3_client.put_object(
            Bucket=model_bucket,
            Key=s3_key,
            Body=model_bytes.getvalue()
        )
        print(f"✓")
        
        s3_path = f"s3://{model_bucket}/{s3_key}"
        print(f"  Model uploaded to: {s3_path}")
        
        return s3_path
        
    except Exception as e:
        raise Exception(f"Failed to upload model to S3: {e}")


def log_model_to_dynamodb(dynamodb_resource, model_table_name, s3_path, job_id):
    """Log model metadata to DynamoDB ModelTable.
    
    Args:
        dynamodb_resource: Boto3 DynamoDB resource
        model_table_name: DynamoDB table name
        s3_path: S3 path of uploaded model
        job_id: Unique identifier for the job

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[DYNAMODB] Logging model to {model_table_name}")
    
    try:
        table = dynamodb_resource.Table(model_table_name)
        item = {
            'model': 'xgbsr',
            'model_job_id': job_id,
            'biweek': f'{year}-BW-{biweek_num}',
            'path': s3_path,
        }
        
        print(f"  Writing model metadata...", end=" ", flush=True)
        table.put_item(Item=item)
        print(f"✓")
        
        return True
        
    except Exception as e:
        print(f"✗")
        print(f"  Error logging to DynamoDB: {e}")
        return False


def log_job_metadata(dynamodb_resource, job_table_name, model_s3_path, job_id, params, start_time, end_time, year, biweek_num):
    """Log job metadata to DynamoDB."""
    
    table = dynamodb_resource.Table(job_table_name)
    item = {
        'job_type': 'xgbsr_training',
        'complete_timestamp': str(end_time)[:-6],
        'job_id': job_id,
        'elapsed_seconds': Decimal(str(f'{(end_time - start_time).total_seconds():.2f}')),
        'biweek': f'{year}-BW-{biweek_num}',
        'model_s3_path': model_s3_path,
        'parameters': params,
    }
    
    print(f"\n[DYNAMODB] Logging job metadata to {job_table_name}")
    
    try:
        print(f"  Writing job metadata...", end=" ", flush=True)
        table.put_item(Item=item)
        print(f"✓")
        
    except Exception as e:
        raise Exception(f"Failed to log job metadata: {e}")


if __name__ == '__main__':
    """
    ECS Fargate entry point for biweekly XGBoost model training.
    
    Args:
        env_name: AWS CDK environment name (e.g., 'dev', 'prod')
    """
    env_name = os.environ.get('ENVIRONMENT', 'dev')

    try:
        print("=" * 70)
        print("Biweekly XGBoost Model Training")
        print("=" * 70)
        
        # Get storage names from CloudFormation exports
        s3_client = boto3.client('s3')
        dynamodb_resource = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
        try:
            print(f"Environment: {env_name}")
            data_bucket_name = os.environ.get('DATA_BUCKET')
            if not data_bucket_name:
                raise ValueError("DATA_BUCKET environment variable not set")
            print(f"Bucket: {data_bucket_name}\n")
            model_bucket_name = os.environ.get('MODEL_BUCKET')
            if not model_bucket_name:
                raise ValueError("MODEL_BUCKET environment variable not set")
            print(f"Model Bucket: {model_bucket_name}\n")
            job_table_name = os.environ.get('JOB_TABLE')
            if not job_table_name:
                raise ValueError("JOB_TABLE environment variable not set")
            print(f"Job Table: {job_table_name}\n")
            model_table_name = os.environ.get('MODEL_TABLE')
            if not model_table_name:
                raise ValueError("MODEL_TABLE environment variable not set")
            print(f"Model Table: {model_table_name}\n")
            year = int(os.environ.get('YEAR'))
            if not year:
                raise ValueError("YEAR environment variable not set")
            print(f"Year: {year}\n")
            biweek_num = int(os.environ.get('BIWEEK_NUM'))
            if not biweek_num:
                raise ValueError("BIWEEK_NUM environment variable not set")
            print(f"Biweek Number: {biweek_num}\n")
        except Exception as e:
            error_msg = f"Failed to retrieve bucket name: {e}"
            print(f"✗ {error_msg}")
            sys.exit(1)

        latest_biweek = f"{year}/BW-{biweek_num}"
        biweek_prefix = f"{latest_biweek}/"

        # Step 1: Load all biweekly SARIMAX prime data
        print("[1/14] Loading all biweekly SARIMAX prime data...")
        prime_data = load_sarimax_prime_data(s3_client, data_bucket_name, year, biweek_num)
        
        # Step 2: Check for marker file indicating processed data is ready
        print(f"\n[2/14] Checking for processed data marker in s3://{data_bucket_name}/{SARIMAX_PRIME_PREFIX}{biweek_prefix}")
        if not marker_exists(s3_client, data_bucket_name, f"{SARIMAX_PRIME_PREFIX}{biweek_prefix}"):
            print(f"✗ Error: Processed SARIMAX prime data is not ready. Marker not found at s3://{data_bucket_name}/{SARIMAX_PRIME_PREFIX}{biweek_prefix}{MARKER}")
            sys.exit(1)
        print(f"✓ Marker found. Processed data is ready.")

        # Step 3: Check for SARIMAX model completion marker
        print(f"\n[3/14] Checking for SARIMAX model completion marker in s3://{model_bucket_name}/{SARIMAX_MODEL_PREFIX}{biweek_prefix}")
        if not marker_exists(s3_client, model_bucket_name, f"{SARIMAX_MODEL_PREFIX}{biweek_prefix}"):
            print(f"✗ Error: SARIMAX models are not ready. Marker not found at s3://{model_bucket_name}/{SARIMAX_MODEL_PREFIX}{biweek_prefix}{MARKER}")
            sys.exit(1)
        print(f"✓ Marker found. SARIMAX models are ready.")

        # Check for XGBoost model completion marker
        print(f"\n[4/14] Checking for existing XGBoost model marker in s3://{model_bucket_name}/{XGBOOST_MODEL_PREFIX}{biweek_prefix}")
        if marker_exists(s3_client, model_bucket_name, f"{XGBOOST_MODEL_PREFIX}{biweek_prefix}"):
            print(f"✗ Error: XGBoost model for {latest_biweek} already exists. Marker found at s3://{model_bucket_name}/{XGBOOST_MODEL_PREFIX}{biweek_prefix}{MARKER}")
            sys.exit(1)
        print(f"✓ No existing XGBoost model marker found. Proceeding with training.")
        
        # Step 4: Download SARIMAX models
        print("\n[5/14] Downloading SARIMAX models...")
        smx_per_family, smx_per_store = download_sarimax_models(s3_client, model_bucket_name, year, biweek_num)
        
        if not smx_per_family or not smx_per_store:
            raise Exception("No SARIMAX models could be loaded")
        
        # Step 5: Load time series per family and store
        print("\n[6/14] Loading time series from S3...")
        ts_per_family, ts_per_store = load_time_series(s3_client, data_bucket_name)
        
        if ts_per_family is None or ts_per_store is None:
            print("✗ Failed to load time series.")
            sys.exit(1)

        # Step 6: Generate SARIMAX predictions
        print("\n[7/14] Generating SARIMAX predictions as features...")
        inf_per_family, inf_per_store = generate_inferences(
                ts_per_family, ts_per_store, smx_per_family, smx_per_store
            )

        # Step 7: Create XGBoost feature dataframe
        print("\n[8/14] Creating XGBoost feature dataframe...")
        X, y, X_cols = create_feature_dataframe(prime_data, inf_per_family, inf_per_store)
        
        if X is None or y is None:
            print("✗ Failed to create feature dataframe.")
            sys.exit(1)

        # Free memory
        print("\n[9/14]Freeing inference, SARIMAX, and time series dictionaries...")
        del inf_per_family
        del inf_per_store
        del ts_per_family
        del ts_per_store
        del smx_per_family
        del smx_per_store
        gc.collect()

        # Step 8: Train stacking model
        print("\n[10/14] Training stacked XGBoost model...")
        sr, xgbsr_params, start_time, end_time = train_stacking_model(X, y)
        
        # Step 9: Save model and upload to S3
        print("\n[11/14] Saving and uploading model to S3...")
        model_bytes = serialize_model(sr, X_cols)
        if model_bytes is None:
            print("✗ Failed to serialize model.")
            sys.exit(1)
        model_s3_path = upload_model_to_s3(s3_client, model_bucket_name, model_bytes, year, biweek_num)
        
        # Step 10: Upload the XGBoost-prime feature data
        print("\n[12/14] Uploading XGBoost-prime feature data to S3...")
        upload_xgboost_prime_data(s3_client, data_bucket_name, prime_data, year, biweek_num)
        
        # Step 11: Log model to DynamoDB
        print("\n[13/14] Logging model metadata to DynamoDB...")
        job_id = str(uuid.uuid4())
        model_logged = log_model_to_dynamodb(dynamodb_resource, model_table_name, model_s3_path, job_id)
        
        if not model_logged:
            print("✗ Failed to log model to DynamoDB.")
            sys.exit(1)

        # Step 12: Log job metadata and write markers
        print("\n[14/14] Logging job metadata...")
        log_job_metadata(
            dynamodb_resource,
            job_table_name,
            model_s3_path,
            job_id,
            xgbsr_params,
            start_time,
            end_time,
            year,
            biweek_num
            )
        
        # Write completion markers
        write_marker(s3_client, data_bucket_name, f"{XGBOOST_PRIME_PREFIX}{biweek_prefix}")
        write_marker(s3_client, model_bucket_name, f"{XGBOOST_MODEL_PREFIX}{biweek_prefix}")
        
        print("\n" + "=" * 70)
        print("✓ Biweekly XGBoost Model Training Completed Successfully!")
        print("=" * 70)
        print(f"Feature data uploaded to: s3://{data_bucket_name}/{XGBOOST_PRIME_PREFIX}{biweek_prefix}")
        print(f"Model uploaded to: {model_s3_path}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        sys.exit(1)
