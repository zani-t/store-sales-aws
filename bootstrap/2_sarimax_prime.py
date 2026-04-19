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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import boto3
from scipy.stats import boxcox
from botocore.exceptions import ClientError

from bootstrap import (
    get_stack_output,
    marker_exists,
    write_marker,
    family_encode
    )

# Dataset names to process
DATASET_NAMES = ['holidays_events', 'oil', 'stores', 'train', 'transactions']
OUTPUT_PREFIX = 'processed/sarimax-prime/historical/'
TIMESERIES_FAMILY_PREFIX = 'processed/sarimax-prime/historical/family/'
TIMESERIES_STORE_PREFIX = 'processed/sarimax-prime/historical/store/'

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

# Create temporary directory for local processing
TEMP_DIR = Path(tempfile.mkdtemp(prefix='sarimax_prime_'))


def download_from_s3(s3_client, bucket_name):
    """Download historical datasets from S3."""
    print(f"\n[S3] Downloading datasets from s3://{bucket_name}/raw/historical/")
    
    datasets = {}
    missing_files = []
    
    for dataset_name in DATASET_NAMES:
        s3_key = f"raw/historical/{dataset_name}.csv"
        local_path = TEMP_DIR / f"{dataset_name}.csv"
        
        try:
            print(f"  Downloading {dataset_name}...", end=" ")
            s3_client.download_file(bucket_name, s3_key, str(local_path))
            
            # Load CSV
            if dataset_name == 'stores':
                datasets[dataset_name] = pd.read_csv(local_path)
            else:
                datasets[dataset_name] = pd.read_csv(local_path, parse_dates=['date'])
            
            print(f"✓ ({len(datasets[dataset_name])} rows)")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                missing_files.append(s3_key)
                print(f"✗ (not found in S3)")
            else:
                print(f"✗")
                print(f"  Error downloading {s3_key}: {error_code}")
                return None
        except Exception as e:
            print(f"✗")
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


def apply_transformations(datasets):
    """Apply SARIMAX Prime transformations to datasets.
    
    Returns:
        tuple: (processed_train_data, lambda_values_dict)
    """
    print(f"\n[TRANSFORM] Applying SARIMAX Prime transformations...")
    
    stores = datasets['stores']
    holidays_events = datasets['holidays_events']
    oil = datasets['oil']
    transactions = datasets['transactions']
    train = datasets['train']
    
    train = train.drop('id', axis=1)
    
    # COMPRESSION
    print("  - Compressing integer and float columns...")
    dataframes = [stores, holidays_events, oil, transactions, train]
    for df in dataframes:
        for col in df:
            if df[col].dtype == 'int64':
                if df[col].max() <= np.iinfo(np.int16).max:
                    if df[col].max() < np.iinfo(np.int8).max:
                        df[col] = df[col].astype('int8')
                    else:
                        df[col] = df[col].astype('int16')
            if df[col].dtype == 'float64':
                if df[col].max() <= np.finfo(np.float32).max:
                    if df[col].max() <= np.finfo(np.float16).max:
                        pass
                    else:
                        df[col] = df[col].astype('float32')
    
    # CONNECTION & IMPUTATION
    print("  - Merging datasets and imputing missing values...")
    oil.rename(columns={'dcoilwtico': 'oilprice'}, inplace=True)
    
    train = train.merge(oil, how='left', on='date')
    train['oilprice'] = train['oilprice'].bfill()
    train = train.merge(transactions, how='left', on=['date', 'store_nbr'])
    train = train.merge(stores, how='left', on='store_nbr')
    train = train.merge(holidays_events, how='left', on='date')
    train.fillna({'transactions': 0}, inplace=True)
    train.rename(columns={'type_x': 'store_type', 'type_y': 'holiday_type'}, inplace=True)
    
    lmbda_sales = boxcox(train.loc[train['sales'] > 0, 'sales'])[1]
    
    # FEATURE ENGINEERING
    print("  - Engineering features...")
    
    # Transferred holidays
    train.loc[train['transferred'] == True, 'holiday_type'] = 'Transferred' + train['holiday_type']
    
    # Active holidays
    train['ntl_holiday'] = (train['locale'] == 'National').astype('int8')
    train['rgnl_holiday'] = ((train['locale'] == 'Regional') & (train['locale_name'] == train['state'])).astype('int8')
    train['lcl_holiday'] = ((train['locale'] == 'Local') & (train['locale_name'] == train['city'])).astype('int8')
    
    # Median transform & zero feature - Onpromotion, Transactions
    median_onpromotion = train.loc[train['onpromotion'] > 0, 'onpromotion'].median()
    train['exists_promotion'] = train['onpromotion'].apply(lambda x: 1 if x > 0 else 0).astype('int8')
    train['onpromotion'] = train['onpromotion'].apply(lambda x: x if x > 0 else median_onpromotion)
    
    median_transactions = transactions.loc[transactions['transactions'] > 0, 'transactions'].median()
    transactions['exists_transaction'] = transactions['transactions'].apply(lambda x: 1 if x > 0 else 0).astype('int8')
    transactions['transactions'] = transactions['transactions'].apply(lambda x: x if x > 0 else median_transactions)
    train = train.drop('transactions', axis=1)
    train = train.merge(transactions, how='left', on=['date', 'store_nbr'])
    train.fillna({'transactions': 0, 'exists_transaction': 0}, inplace=True)
    
    # PriceStatus, LowPrice, HighPrice
    cutoff = 71.5
    oil['oil_price_status'] = oil['oilprice'].apply(lambda x: 1 if x < cutoff else 0)
    median_lowoilprice = oil.loc[((oil['date'] < '2017-08-15') & (oil['oilprice'] <= cutoff)), 'oilprice'].median()
    oil['low_oil_price'] = oil['oilprice'].apply(lambda x: x if x <= cutoff else median_lowoilprice)
    median_highoilprice = oil.loc[((oil['date'] < '2017-08-15') & (oil['oilprice'] > cutoff)), 'oilprice'].median()
    oil['high_oil_price'] = oil['oilprice'].apply(lambda x: x if x > cutoff else median_highoilprice)
    oil = oil.drop('oilprice', axis=1)
    
    train = train.drop('oilprice', axis=1)
    train = train.merge(oil, how='left', on=['date'])
    train['oil_price_status'] = train['oil_price_status'].bfill()
    train['low_oil_price'] = train['low_oil_price'].bfill()
    train['high_oil_price'] = train['high_oil_price'].bfill()
    
    # BoxCox transforms
    print("  - Applying BoxCox transforms...")
    lmbda_onpromotion = boxcox(train.loc[train['onpromotion'] > 0, 'onpromotion'])[1]
    lmbda_transactions = boxcox(transactions.loc[transactions['transactions'] > 0, 'transactions'])[1]
    train['onpromotion'] = boxcox(train['onpromotion'] + 0.01, lmbda_onpromotion)
    train['transactions'] = boxcox(train['transactions'] + 0.01, lmbda_transactions)
    train['sales'] = boxcox(train['sales'] + 0.01, lmbda_sales)
    
    # Target encoding - HolidayMeanVariation
    print("  - Computing holiday mean variations...")
    ma = train[['date', 'sales']].groupby(['date']).agg({'sales': 'mean'})
    ma = pd.DataFrame(ma.rolling(window=15, min_periods=15).mean().values, columns=['ma30']).set_index(ma.index)
    train = train.merge(ma, how='left', on='date')
    train['hmv'] = 0.0
    hmvs = {}
    for holiday in holidays_events['description'].unique():
        df = train.loc[train['description'] == holiday, ['date', 'ma30', 'sales']].groupby(['date', 'ma30'], as_index=False).agg(sales=('sales', 'mean'))
        hmv = (df['sales'] - df['ma30']).mean()
        hmvs[holiday] = float(hmv)
        train.loc[train['description'] == holiday, 'hmv'] = ((train['ntl_holiday'] == 1) |
                                                             (train['rgnl_holiday'] == 1) |
                                                             (train['lcl_holiday'] == 1)).astype('int8') * hmv
    
    # One-hot encoding
    print("  - One-hot encoding categorical features...")
    train.loc[(train['ntl_holiday'] == 0) & (train['rgnl_holiday'] == 0) & (train['lcl_holiday'] == 0), 'holiday_type'] = np.nan
    train = pd.get_dummies(train, columns=['holiday_type', 'store_type'])
    
    cols_to_int = ['holiday_type_Additional', 'holiday_type_Bridge', 'holiday_type_Event', 'holiday_type_Holiday',
                   'holiday_type_Transfer', 'holiday_type_TransferredHoliday', 'holiday_type_Work Day', 'store_type_A',
                   'store_type_B', 'store_type_C', 'store_type_D', 'store_type_E']
    train[cols_to_int] = train[cols_to_int].astype('int8')
    
    train = train.drop(['locale', 'locale_name', 'description', 'transferred', 'ma30'], axis=1)
    
    # Store lambda values for inverse transforms
    lambdas = {
        'lmbda_sales': float(lmbda_sales),
        'lmbda_onpromotion': float(lmbda_onpromotion),
        'lmbda_transactions': float(lmbda_transactions)
    }
    
    print(f"\n✓ Transformations complete. Final dataset: {len(train)} rows")
    return train, lambdas, hmvs


def upload_to_s3(s3_client, bucket_name, processed_data, lambdas, hmvs):
    """Upload processed parquet file and lambda values to S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        processed_data: Processed DataFrame
        lambdas: Dictionary with lambda transform values
    """
    print(f"\n[S3] Uploading processed data to s3://{bucket_name}/{OUTPUT_PREFIX}")

    def save_json_to_s3(data, s3_key):
        buffer = BytesIO()
        json_data = json.dumps(data, indent=2)
        buffer.write(json_data.encode('utf-8'))
        buffer.seek(0)
        
        print(f"  Uploading {s3_key}...", end=" ")
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        print(f"✓")
    
    try:
        # Upload parquet file
        parquet_buffer = BytesIO()
        processed_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_key = f"{OUTPUT_PREFIX}data.parquet"
        print(f"  Uploading {s3_key}...", end=" ")
        s3_client.upload_fileobj(parquet_buffer, bucket_name, s3_key)
        print(f"✓")
        
        # Upload lambda values as JSON
        save_json_to_s3(lambdas, f"{OUTPUT_PREFIX}lambdas.json")

        # Upload HMV values as JSON
        save_json_to_s3(hmvs, f"{OUTPUT_PREFIX}hmvs.json")

        # Upload families filename mapping as JSON
        family_mapping = {family_encode(k): k for k in (NON_TWO_YEAR_FAMILIES.keys() | TWO_YEAR_FAMILIES)}
        save_json_to_s3(family_mapping, f"{OUTPUT_PREFIX}families_mapping.json")
        
        print(f"\n✓ Successfully uploaded processed data to S3")
        print(f"  Bucket: {bucket_name}")
        print(f"  Path: s3://{bucket_name}/{OUTPUT_PREFIX}")
        
        return True
        
    except ClientError as e:
        print(f"✗")
        print(f"  Error uploading to S3: {e}")
        return False
    except Exception as e:
        print(f"✗")
        print(f"  Error: {e}")
        return False


def build_and_upload_time_series(s3_client, bucket_name, processed_data):
    """Build time series aggregations and upload to S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        processed_data: Processed training DataFrame
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[TIMESERIES] Building aggregated time series per family and store")
    
    ts_per_family = {}
    ts_per_store = {}
    
    try:
        # Build time series per family
        print(f"  Building time series per family...")
        for f in processed_data['family'].unique():
            if f in NON_TWO_YEAR_FAMILIES:
                ts_per_family[f] = processed_data.loc[
                    (processed_data['date'] > PERIOD_MAP[NON_TWO_YEAR_FAMILIES[f]]) &
                    (processed_data['family'] == f)
                ].groupby(['date']).agg(EXOG_FEATURES)
            elif f in TWO_YEAR_FAMILIES:
                ts_per_family[f] = processed_data.loc[
                    (processed_data['date'] > '2015-08-15') & (processed_data['family'] == f)
                ].groupby(['date']).agg(EXOG_FEATURES)
        print(f"    ✓ Built time series for {len(ts_per_family)} families")
        
        # Build time series per store
        print(f"  Building time series per store...")
        for s in range(1, 55):
            if s in NON_TWO_YEAR_STORES:
                store_data = processed_data.loc[
                    (processed_data['date'] > PERIOD_MAP[NON_TWO_YEAR_STORES[s]]) &
                    (processed_data['store_nbr'] == s)
                ].groupby(['date']).agg(EXOG_FEATURES)
            elif s in TWO_YEAR_STORES:
                store_data = processed_data.loc[
                    (processed_data['date'] > '2015-08-15') & (processed_data['store_nbr'] == s)
                ].groupby(['date']).agg(EXOG_FEATURES)
            else:
                continue
            
            if len(store_data) > 0:
                ts_per_store[s] = store_data
        print(f"    ✓ Built time series for {len(ts_per_store)} stores")
        
        # Upload family time series
        print(f"\n[S3] Uploading family time series to s3://{bucket_name}/{TIMESERIES_FAMILY_PREFIX}")
        for f, ts_data in ts_per_family.items():
            try:
                parquet_buffer = BytesIO()
                ts_data.to_parquet(parquet_buffer, index=True)
                parquet_buffer.seek(0)
                
                s3_key = f"{TIMESERIES_FAMILY_PREFIX}{family_encode(f)}.parquet"
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=parquet_buffer.getvalue()
                )
            except Exception as e:
                print(f"    ✗ Error uploading family '{f}': {e}")
                return False
        print(f"  ✓ Uploaded {len(ts_per_family)} family time series")
        
        # Upload store time series
        print(f"\n[S3] Uploading store time series to s3://{bucket_name}/{TIMESERIES_STORE_PREFIX}")
        for s, ts_data in ts_per_store.items():
            try:
                parquet_buffer = BytesIO()
                ts_data.to_parquet(parquet_buffer, index=True)
                parquet_buffer.seek(0)
                
                s3_key = f"{TIMESERIES_STORE_PREFIX}store_{s:02d}.parquet"
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=parquet_buffer.getvalue()
                )
            except Exception as e:
                print(f"    ✗ Error uploading store {s}: {e}")
                return False
        print(f"  ✓ Uploaded {len(ts_per_store)} store time series")
        
        print(f"\n✓ Successfully uploaded time series to S3")
        return True
        
    except Exception as e:
        print(f"  ✗ Error building time series: {e}")
        return False


def cleanup_temp_files():
    """Remove temporary local directory."""
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
    # Get S3 bucket name from CDK CloudFormation exports
    s3_bucket_name = get_stack_output(env_name, f"{env_name}-DataBucketName")
    
    print("=" * 60)
    print("SARIMAX Prime Data Processing")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"S3 Bucket: {s3_bucket_name}")
    print(f"Temp Directory: {TEMP_DIR}")
    print()
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Check for marker file indicating raw data is ready
        print("[1/7] Checking for historical data marker...")
        if not marker_exists(s3_bucket_name, "raw/historical/"):
            print("✗ Error: Marker file not found at s3://{}/raw/historical/_COMPLETE".format(s3_bucket_name))
            print("  The raw data has not been successfully uploaded.")
            print("  Please run: python bootstrap/1_raw.py {}".format(env_name))
            sys.exit(1)
        print("✓ Marker found. Raw data is ready for processing.")
        
        # Check if processing has already been completed
        print("\n[2/7] Checking if processing has already been completed...")
        if marker_exists(s3_bucket_name, OUTPUT_PREFIX):
            print("✗ Error: Processing workflow has already been completed.")
            sys.exit(0)
        print("✓ No marker found. Ready to proceed with processing.")
        
        # Download from S3
        print("\n[3/7] Downloading historical data from S3...")
        datasets = download_from_s3(s3_client, s3_bucket_name)
        
        if datasets is None:
            print("Error: Failed to download datasets from S3")
            sys.exit(1)
        
        # Apply transformations
        print("\n[4/7] Applying SARIMAX Prime transformations...")
        processed_data, lambdas, hmvs = apply_transformations(datasets)
        
        # Upload to S3
        print("\n[5/7] Uploading processed data to S3...")
        upload_success = upload_to_s3(s3_client, s3_bucket_name, processed_data, lambdas, hmvs)
        
        if not upload_success:
            print("Error: Failed to upload processed data to S3")
            sys.exit(1)
        
        # Build and upload time series
        print("\n[6/7] Building and uploading time series...")
        ts_success = build_and_upload_time_series(s3_client, s3_bucket_name, processed_data)
        
        if not ts_success:
            print("Error: Failed to build and upload time series")
            sys.exit(1)
        
        # Write marker and cleanup
        print("\n[7/7] Finalizing...")
        write_marker(s3_bucket_name, OUTPUT_PREFIX)
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python 2_sarimax_prime.py <env_name>")
        print("Example: python 2_sarimax_prime.py dev")
        sys.exit(1)
    
    env_name = sys.argv[1]
    main(env_name)


