#!/usr/bin/env python3
"""
Train SARIMAX models per family and per store on biweekly data.
Downloads subprime data from S3, applies prime transformations, trains models,
saves them to S3. Models are stored in <model_bucket>/sarimax/biweekly/<year>/BW-<biweek>/.
"""

import os
import sys
import json
import pickle
import shutil
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
import numpy as np
from scipy.stats import boxcox
from botocore.exceptions import ClientError
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

# Suppress warnings
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Constants
MARKER = '_COMPLETE'
SUBPRIME_INPUT_PREFIX = 'processed/sarimax-subprime/biweekly/'
PRIME_INPUT_PREFIX = 'processed/sarimax-prime/biweekly/'
PRIME_OUTPUT_PREFIX = 'processed/sarimax-prime/biweekly/'
TIMESERIES_FAMILY_PREFIX = 'processed/sarimax-prime/biweekly/'
TIMESERIES_STORE_PREFIX = 'processed/sarimax-prime/biweekly/'
SARIMAX_OUTPUT_PREFIX = 'sarimax/biweekly/'
SIGNIFICANT_EXOG = ['hmv', 'exists_promotion', 'exists_transaction']

# Time series construction parameters
PERIOD_MAP = {
    0.25: '2017-05-15',
    0.5: '2017-02-15',
    1: '2016-08-15',
    1.5: '2016-02-15',
    2.5: '2015-02-15',
    3.5: '2014-02-15',
    4: '2013-01-01'
}

NON_TWO_YEAR_FAMILIES = {
    'BABY CARE': 1.5, 'BOOKS': 0.5, 'LAWN AND GARDEN': 0.5, 'LIQUOR,WINE,BEER': 1,
    'MAGAZINES': 1.5, 'AUTOMOTIVE': 4, 'BEAUTY': 4, 'BREAD/BAKERY': 4, 'CLEANING': 4,
    'DAIRY': 3.5, 'DELI': 4, 'EGGS': 4, 'FROZEN FOODS': 4, 'GROCERY I': 4,
    'GROCERY II': 4, 'LINGERIE': 4, 'MEATS': 4, 'PERSONAL CARE': 4, 'POULTRY': 3.5,
    'PREPARED FOODS': 4, 'SEAFOOD': 2.5, 'SCHOOL AND OFFICE SUPPLIES': 1
}

TWO_YEAR_FAMILIES = {
    'BEVERAGES', 'CELEBRATION', 'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II',
    'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'PRODUCE'
}

NON_TWO_YEAR_STORES = {21: 1, 22: 1.5, 25: 0.5, 42: 1.5, 52: 0.25, 53: 1}
TWO_YEAR_STORES = {*range(1, 21), 23, 24, *range(26, 42), *range(43, 52), 54}

EXOG_FEATURES = {feature: 'mean' for feature in [
    'sales', 'onpromotion', 'transactions', 'ntl_holiday', 'rgnl_holiday', 'lcl_holiday', 'hmv', 'exists_promotion',
    'exists_transaction', 'oil_price_status', 'low_oil_price', 'high_oil_price', 'holiday_type_Additional',
    'holiday_type_Bridge', 'holiday_type_Event', 'holiday_type_Holiday', 'holiday_type_Transfer',
    'holiday_type_TransferredHoliday', 'holiday_type_Work Day'
]}

# Temporary directory for local model storage
TEMP_DIR = Path(tempfile.mkdtemp(prefix='sarimax_biweekly_'))
FAMILY_DIR = TEMP_DIR / 'family'
STORE_DIR = TEMP_DIR / 'store'


def get_stack_output(env_name, export_name):
    """Get CloudFormation stack output by export name."""
    try:
        cf = boto3.client('cloudformation')
        response = cf.describe_stacks(StackName=f"{env_name}-StorageStack")
        outputs = response['Stacks'][0]['Outputs']
        return next(o['OutputValue'] for o in outputs if o['ExportName'] == export_name)
    except Exception as e:
        raise Exception(f"Failed to get stack output '{export_name}': {e}")


def marker_exists(s3_client, bucket, prefix):
    """Check if completion marker exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}{MARKER}")
        return True
    except ClientError:
        return False


def write_marker(s3_client, bucket, prefix):
    """Write completion marker to S3."""
    s3_client.put_object(Bucket=bucket, Key=f"{prefix}{MARKER}", Body=b'')


def family_encode(name):
    """Encode family name for use in filenames."""
    return name.replace(' ', '_').replace('/', '_')


def ensure_directories_exist():
    """Create temporary output directories."""
    print(f"\n[SETUP] Creating temporary directories")
    
    for directory in [FAMILY_DIR, STORE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def load_subprime_data(s3_client, bucket_name):
    """Load all biweekly and historical subprime data from S3 and concatenate into one dataset.
    
    Returns:
        tuple: (combined_data DataFrame, latest_year, latest_biweek)
    """
    try:
        all_data = []
        latest_year = None
        latest_biweek = None
        
        # Step 1: Load historical data
        print(f"\n[S3] Loading historical subprime data...")
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key='processed/sarimax-subprime/historical/data.parquet'
            )
            hist_data = pd.read_parquet(BytesIO(response['Body'].read()))
            all_data.append(hist_data)
            print(f"  ✓ Loaded historical data: {len(hist_data)} rows")
        except ClientError as e:
            print(f"  ⚠ Could not load historical data: {e}")
        
        # Step 2: Discover all biweekly folders
        print(f"\n[S3] Discovering all biweekly folders...")
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
        
        # Step 3: Load all biweekly data
        biweek_count = 0
        for year_folder in year_folders:
            # Extract year from path
            year = int(year_folder.rstrip('/').split('/')[-1])
            
            # Get all biweek folders for this year
            pages = paginator.paginate(Bucket=bucket_name, Prefix=year_folder, Delimiter='/')
            for page in pages:
                for prefix in page.get('CommonPrefixes', []):
                    biweek_folder = prefix['Prefix']
                    biweek_str = biweek_folder.rstrip('/').split('/')[-1]
                    
                    if biweek_str.startswith('BW-'):
                        try:
                            biweek_num = int(biweek_str.replace('BW-', ''))
                            
                            # Load this biweek's data
                            response = s3_client.get_object(
                                Bucket=bucket_name,
                                Key=f'{SUBPRIME_INPUT_PREFIX}{year}/BW-{biweek_num}/data.parquet'
                            )
                            bw_data = pd.read_parquet(BytesIO(response['Body'].read()))
                            all_data.append(bw_data)
                            biweek_count += 1
                            
                            # Track latest biweek
                            if (latest_biweek is None) or (latest_year is None) or \
                                (year > latest_year) or (year == latest_year and biweek_num > latest_biweek):
                                latest_year = year
                                latest_biweek = biweek_num
                            
                            print(f"  ✓ Loaded {year}/BW-{biweek_num}: {len(bw_data)} rows")
                        except Exception as e:
                            print(f"  ⚠ Could not load {year}/{biweek_str}: {e}")
        
        if not all_data:
            raise Exception("No subprime data could be loaded from S3")
        
        # Step 4: Concatenate all data
        print(f"\n[DATA] Concatenating {biweek_count} biweekly dataset(s) + historical data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"  ✓ Combined dataset: {len(combined_data)} rows, {len(combined_data.columns)} columns")
        
        if latest_year is None or latest_biweek is None:
            raise Exception("Could not determine latest biweek from loaded data")
        
        return combined_data, latest_year, latest_biweek
        
    except Exception as e:
        raise Exception(f"Failed to load subprime data: {e}")


def load_families_mapping(s3_client, bucket_name):
    """Load families mapping from S3."""
    try:
        print(f"\n[S3] Loading families mapping...")
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key="processed/sarimax-prime/historical/families_mapping.json"
            )
            families = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded families mapping for {len(families)} families")
        except Exception as e:
            print(f"  ✗ Could not load families_mapping.json: {e}")
            raise
        
        return families
    except Exception as e:
        raise Exception(f"Failed to load jsons from S3: {e}")


def prime_data_for_sarimax(data):
    """Apply Box-Cox transformations and fill HMV values in preparation for SARIMAX modeling."""
    try:
        print("\n[TRANSFORM] Applying prime transformations for SARIMAX...")

        # Apply Box-Cox transformations
        lmbda_sales = boxcox(data.loc[data['sales'] > 0, 'sales'])[1]
        lmbda_onpromotion = boxcox(data.loc[data['onpromotion'] > 0, 'onpromotion'])[1]
        lmbda_transactions = boxcox(data.loc[data['transactions'] > 0, 'transactions'])[1]

        data['onpromotion'] = boxcox(data['onpromotion'] + 0.01, lmbda_onpromotion)
        data['transactions'] = boxcox(data['transactions'] + 0.01, lmbda_transactions)
        data['sales'] = boxcox(data['sales'] + 0.01, lmbda_sales)

        # Fill HMVs
        print("  - Computing HMV values...")
        ma = data[['date', 'sales']].groupby(['date']).agg({'sales': 'mean'})
        ma = pd.DataFrame(ma.rolling(window=15, min_periods=1).mean().values, columns=['ma15']).set_index(ma.index)
        data = data.merge(ma, how='left', on='date')
        data['hmv'] = 0.0
        hmvs = {}
        for holiday in data['description'].unique():
            df = data.loc[data['description'] == holiday, ['date', 'ma15', 'sales']].groupby(
                ['date', 'ma15'], as_index=False
            ).agg(sales=('sales', 'mean'))
            hmv = (df['sales'] - df['ma15']).mean()
            hmvs[holiday] = float(hmv)
            data.loc[data['description'] == holiday, 'hmv'] = (
                ((data['ntl_holiday'] == 1) | (data['rgnl_holiday'] == 1) | (data['lcl_holiday'] == 1)).astype('int8') * hmv
            )
        
        data = data.drop(['description', 'ma15'], axis=1)
        
       # Store lambda values for inverse transforms
        lambdas = {
            'lmbda_sales': float(lmbda_sales),
            'lmbda_onpromotion': float(lmbda_onpromotion),
            'lmbda_transactions': float(lmbda_transactions)
        }
        
        print(f"✓ Prime transformations complete. Final dataset: {len(data)} rows")
        return data, lambdas, hmvs
    except Exception as e:
        raise Exception(f"Error during prime transformations: {e}")


def build_time_series(data):
    """Aggregate data to build time series for SARIMAX modeling."""
    print(f"\n[TIMESERIES] Building aggregated time series per family and store")
    
    ts_per_family = {}
    ts_per_store = {}

    try:
        # Build time series per family
        print(f"  Building time series per family...")
        for f in data['family'].unique():
            if f in NON_TWO_YEAR_FAMILIES:
                ts_per_family[f] = data.loc[
                    (data['date'] > PERIOD_MAP[NON_TWO_YEAR_FAMILIES[f]]) &
                    (data['family'] == f)
                ].groupby(['date']).agg(EXOG_FEATURES)
            elif f in TWO_YEAR_FAMILIES:
                ts_per_family[f] = data.loc[
                    (data['date'] > '2015-08-15') & (data['family'] == f)
                ].groupby(['date']).agg(EXOG_FEATURES)
        print(f"    ✓ Built time series for {len(ts_per_family)} families")
        
        # Build time series per store
        print(f"  Building time series per store...")
        for s in range(1, 55):
            if s in NON_TWO_YEAR_STORES:
                store_data = data.loc[
                    (data['date'] > PERIOD_MAP[NON_TWO_YEAR_STORES[s]]) &
                    (data['store_nbr'] == s)
                ].groupby(['date']).agg(EXOG_FEATURES)
            elif s in TWO_YEAR_STORES:
                store_data = data.loc[
                    (data['date'] > '2015-08-15') & (data['store_nbr'] == s)
                ].groupby(['date']).agg(EXOG_FEATURES)
            else:
                continue
            
            if len(store_data) > 0:
                ts_per_store[s] = store_data
        print(f"    ✓ Built time series for {len(ts_per_store)} stores")
        
        return ts_per_family, ts_per_store
        
    except Exception as e:
        raise Exception(f"Error building time series: {e}")


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
        'biweek': f'BW-{biweek_num}',
        'year': year,
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
    
    if not data_bucket_name:
        print("Error: DATA_BUCKET environment variable not set")
        sys.exit(1)
    if not model_bucket_name:
        print("Error: MODEL_BUCKET environment variable not set")
        sys.exit(1)
    if not job_table_name:
        print("Error: JOB_TABLE environment variable not set")
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
        
        # Step 1: Load all biweekly and historical subprime data
        print("[1/8] Loading all biweekly and historical subprime data...")
        subprime_data, year, biweek_num = load_subprime_data(s3_client, data_bucket_name)
        
        if year is None or biweek_num is None:
            print("✗ Error: Could not determine latest biweek from loaded data")
            sys.exit(1)
        
        print(f"✓ Latest biweek: {year}/BW-{biweek_num}")
        
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
        prime_data, lambdas, hmvs = prime_data_for_sarimax(subprime_data)
        
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
