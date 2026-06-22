#!/usr/bin/env python3
"""
Train and deploy stacked XGBoost model ensemble.
Loads prepared feature data from S3, trains a stacking ensemble of XGBoost models,
serializes the model, uploads to S3, and logs metadata to DynamoDB.
Output model is stored in <model_bucket>/xgboost/historical/model.joblib.
"""
import os
import sys
import uuid
import datetime
from datetime import datetime as dt
from pathlib import Path
from io import BytesIO
from decimal import Decimal

import boto3
import numpy as np
import pandas as pd
import joblib
from botocore.exceptions import ClientError

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bootstrap import get_stack_output, marker_exists, write_marker

# S3 paths
S3_INPUT_PREFIX = 'processed/xgboost-prime/historical/'
S3_OUTPUT_PREFIX = 'xgboost/historical/'
MODEL_FILENAME = 'model.joblib'


def download_data_from_s3(s3_client, bucket_name):
    """Download processed parquet file from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 data bucket name
    
    Returns:
        tuple: (X DataFrame, y DataFrame) or (None, None) if error
    """
    print(f"\n[S3] Downloading processed data from s3://{bucket_name}/{S3_INPUT_PREFIX}")
    
    s3_key = f"{S3_INPUT_PREFIX}data.parquet"
    
    try:
        print(f"  Downloading {s3_key}...", end=" ", flush=True)
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        data_bytes = BytesIO(response['Body'].read())
        
        X_full = pd.read_parquet(data_bytes)
        print(f"✓")
        print(f"  Loaded {len(X_full)} rows, {len(X_full.columns)} columns")
        
        # Extract target variable if present
        if 'sales' in X_full.columns:
            y = X_full.pop('sales').to_frame()
            X = X_full
            X_cols = X.columns.tolist()
            print(f"  Target variable (sales) extracted from data")
            return X, y, X_cols
        else:
            print(f"  Warning: 'sales' column not found.")
            X_cols = X_full.columns.tolist()
            return X_full, None, X_cols
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"✗")
            print(f"  File not found: s3://{bucket_name}/{s3_key}")
        else:
            print(f"✗")
            print(f"  Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return None, None, None


def train_stacking_model(X, y):
    """Train stacked XGBoost regression model.
    
    Args:
        X: DataFrame with features
        y: DataFrame or Series with target variable
    
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
        print(f"  Input shape: X={X.shape}, y={y.shape if hasattr(y, 'shape') else len(y)}")
        
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
        print(f"  ✗ Error training model: {e}")
        return None, None, None, None


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


def upload_model_to_s3(s3_client, model_bucket, model_bytes):
    """Upload serialized model to S3.
    
    Args:
        s3_client: Boto3 S3 client
        model_bucket: S3 model bucket name
        model_bytes: BytesIO with serialized model
    
    Returns:
        str: S3 path of uploaded model, or None if error
    """
    print(f"\n[S3] Uploading model to s3://{model_bucket}/{S3_OUTPUT_PREFIX}")
    
    try:
        s3_key = f"{S3_OUTPUT_PREFIX}{MODEL_FILENAME}"
        
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
        
    except ClientError as e:
        print(f"✗")
        print(f"  Error uploading to S3: {e}")
        return None
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return None


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
            'biweek': 'historical',
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


def log_job_to_dynamodb(dynamodb_resource, job_table_name, s3_path, job_id, params, start_time, end_time):
    """Log job metadata to DynamoDB JobTable.
    
    Args:
        dynamodb_resource: Boto3 DynamoDB resource
        job_table_name: DynamoDB table name
        s3_path: S3 path of uploaded model
        job_id: Unique identifier for the job
        started_at: Job start timestamp (ISO format string)
        finished_at: Job end timestamp (ISO format string)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[DYNAMODB] Logging job to {job_table_name}")
    
    try:
        table = dynamodb_resource.Table(job_table_name)
        item = {
            'job_type': 'bootstrap_xgbsr',
            'complete_timestamp': str(end_time)[:-6],
            'job_id': job_id,
            'elapsed_seconds': Decimal(str(f'{(end_time - start_time).total_seconds():.2f}')),
            'biweek': 'historical',
            'model_s3_path': s3_path,
            'parameters': params,
        }
        
        print(f"  Writing job metadata...", end=" ", flush=True)
        table.put_item(Item=item)
        print(f"✓")
        
        return True
        
    except Exception as e:
        print(f"✗")
        print(f"  Error logging to DynamoDB: {e}")
        return False


def main(env_name):
    """Main entry point.
    
    Args:
        env_name: AWS CDK environment name (e.g., 'dev', 'prod')
    """
    # Get S3 and DynamoDB resource names from CDK CloudFormation exports
    try:
        data_bucket_name = get_stack_output(env_name, f"{env_name}-DataBucketName")
        model_bucket_name = get_stack_output(env_name, f"{env_name}-ModelBucketName")
        model_table_name = get_stack_output(env_name, f"{env_name}-ModelTableName")
        job_table_name = get_stack_output(env_name, f"{env_name}-JobTableName")
    except Exception as e:
        print(f"Error retrieving resource names: {e}")
        print(f"Please ensure CDK stack '{env_name}-StorageStack' exists.")
        sys.exit(1)
    
    print("=" * 60)
    print("XGBoost Model Training")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Data Bucket: {data_bucket_name}")
    print(f"Model Bucket: {model_bucket_name}")
    print(f"Model Table: {model_table_name}")
    print(f"Job Table: {job_table_name}")
    print()
    
    try:
        # Initialize boto3 clients
        s3_client = boto3.client('s3')
        dynamodb_resource = boto3.resource('dynamodb')
        
        # Check for input data marker
        print("\n[1/6] Checking for input data marker...")
        if not marker_exists(data_bucket_name, S3_INPUT_PREFIX):
            print("✗ Input data marker not found.")
            print("  Please ensure 4_xgboost_prime.py has been run first.")
            sys.exit(1)
        print("✓ Marker found. Input data is ready.")
        
        # Check if model training has already been completed
        print("\n[2/6] Checking if model training has already been completed...")
        if marker_exists(model_bucket_name, S3_OUTPUT_PREFIX):
            print("✓ Marker found. Model training already completed.")
            print(f"  Output: s3://{model_bucket_name}/{S3_OUTPUT_PREFIX}")
            return
        print("✓ No marker found. Ready to proceed with training.")
        
        # Download data
        print("\n[3/6] Downloading training data...")
        X, y, X_cols = download_data_from_s3(s3_client, data_bucket_name)
        
        if X is None or y is None:
            print("✗ Failed to download training data.")
            sys.exit(1)
        
        # Train model
        print("\n[4/6] Training stacked XGBoost model...")
        sr, xgbsr_params, start_time, end_time = train_stacking_model(X, y)
        
        if sr is None:
            print("✗ Failed to train model.")
            sys.exit(1)
        
        # Serialize model
        print("\n[5/6] Serializing model...")
        model_bytes = serialize_model(sr, X_cols)
        
        if model_bytes is None:
            print("✗ Failed to serialize model.")
            sys.exit(1)
        
        # Upload model to S3
        print("\n[6/6] Uploading model and logging metadata...")
        s3_path = upload_model_to_s3(s3_client, model_bucket_name, model_bytes)
        
        if s3_path is None:
            print("✗ Failed to upload model to S3.")
            sys.exit(1)
        
        # Log model to DynamoDB
        job_id = str(uuid.uuid4())
        model_logged = log_model_to_dynamodb(dynamodb_resource, model_table_name, s3_path, job_id)
        
        if not model_logged:
            print("✗ Failed to log model to DynamoDB.")
            sys.exit(1)
        
        # Record end time and log job
        job_logged = log_job_to_dynamodb(
            dynamodb_resource,
            job_table_name,
            s3_path,
            job_id,
            xgbsr_params,
            start_time,
            end_time
        )
        
        if not job_logged:
            print("✗ Failed to log job to DynamoDB.")
            sys.exit(1)
        
        # Write marker and cleanup
        write_marker(model_bucket_name, S3_OUTPUT_PREFIX)
        
        print("\n" + "=" * 60)
        print("✓ XGBoost model training completed successfully!")
        print("=" * 60)
        print(f"Output: {s3_path}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 5_xgboost.py <env_name>")
        print("Example: python 5_xgboost.py dev")
        sys.exit(1)
    
    env_name = sys.argv[1]
    main(env_name)

