import os
import json
from enum import Enum
from io import BytesIO
from datetime import datetime, timedelta
from calendar import monthrange

import boto3
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from botocore.exceptions import ClientError

MARKER = '_COMPLETE'

# Configuration
BIWEEKLY_INPUT_PREFIX = 'processed/sarimax-prime/biweekly/'
BIWEEKLY_OUTPUT_PREFIX = 'processed/sarimax-subprime/biweekly/'
BIWEEKLY_DATASET_NAMES = ['holidays_events', 'oil', 'train', 'transactions']


class IO(Enum):
    INPUT = 1
    OUTPUT = 2


def get_full_biweekly_prefix(year, biweek_num, io_type):
    if io_type == IO.INPUT:
        return f"{BIWEEKLY_INPUT_PREFIX}{year}/BW-{biweek_num}/"
    elif io_type == IO.OUTPUT:
        return f"{BIWEEKLY_OUTPUT_PREFIX}{year}/BW-{biweek_num}/"


def marker_exists(s3_client, bucket, prefix):
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}{MARKER}")
        return True
    except ClientError:
        return False


def write_marker(s3_client, bucket, prefix):
    s3_client.put_object(Bucket=bucket, Key=f"{prefix}{MARKER}", Body=b'')


def load_daily_csvs(s3_client, bucket_name, biweek_start, biweek_end):
    """Load and concatenate CSV files from all days in biweek period.
    
    Returns:
        dict: {dataset_name: concatenated_dataframe}
    """
    print(f"[S3] Loading daily CSVs from {biweek_start.date()} to {biweek_end.date()}")
    
    # Generate all dates in the biweek period
    current_date = biweek_start
    datasets = {name: [] for name in BIWEEKLY_DATASET_NAMES}
    
    while current_date <= biweek_end:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        date_str = f'{year}/{month:02d}/{day:02d}'
        folder_prefix = f"raw/daily/{date_str}/"
        
        for dataset_name in BIWEEKLY_DATASET_NAMES:
            s3_key = f"{folder_prefix}{dataset_name}.csv"
            
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                df = pd.read_csv(BytesIO(response['Body'].read()))
                
                # Parse date column if applicable
                if dataset_name != 'stores' and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                datasets[dataset_name].append(df)
                print(f"  ✓ Loaded {dataset_name} from {date_str}")
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    print(f"  ⚠ Error loading {dataset_name} from {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    # Concatenate all dataframes for each dataset
    concatenated = {}
    for dataset_name, dfs in datasets.items():
        if dfs:
            concatenated[dataset_name] = pd.concat(dfs, ignore_index=True)
            # Remove duplicates if any
            if dataset_name != 'stores':
                concatenated[dataset_name] = concatenated[dataset_name].drop_duplicates()
            print(f"  ✓ Concatenated {dataset_name}: {len(concatenated[dataset_name])} rows")
        else:
            print(f"  ✗ No data found for {dataset_name}")
            return None
    
    return concatenated


def load_stores_csv(s3_client, bucket_name):
    """Load stores.csv from S3 (used for feature engineering).
    
    Returns:
        DataFrame or None if error
    """
    s3_key = "raw/historical/stores.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_csv(BytesIO(response['Body'].read()))
        print(f"[S3] Loaded stores.csv with {len(df)} rows")
        return df
    except ClientError as e:
        print(f"Error loading stores.csv: {e}")
        return None


def apply_sarimax_prime_transforms(datasets):
    """Apply SARIMAX Prime transformations.
    
    Returns:
        DataFrame: processed_dataframe
    """
    print("\n[TRANSFORM] Applying SARIMAX Prime transformations...")
    
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
    train = train.merge(oil, how='left', on='date')
    train['oil_price_status'] = train['oil_price_status'].bfill()
    train['low_oil_price'] = train['low_oil_price'].bfill()
    train['high_oil_price'] = train['high_oil_price'].bfill()
    
    # One-hot encoding
    print("  - One-hot encoding categorical features...")
    train.loc[(train['ntl_holiday'] == 0) & (train['rgnl_holiday'] == 0) & (train['lcl_holiday'] == 0), 'holiday_type'] = np.nan
    train = pd.get_dummies(train, columns=['holiday_type', 'store_type'])
    
    cols_to_int = ['holiday_type_Additional', 'holiday_type_Bridge', 'holiday_type_Event', 'holiday_type_Holiday',
                   'holiday_type_Transfer', 'holiday_type_TransferredHoliday', 'holiday_type_Work Day', 'store_type_A',
                   'store_type_B', 'store_type_C', 'store_type_D', 'store_type_E']
    
    train = train.reindex(columns=train.columns.union(cols_to_int), fill_value=0)
    train[cols_to_int] = train[cols_to_int].astype('int8')
    
    train = train.drop(['locale', 'locale_name', 'transferred'], axis=1)
    
    print(f"✓ Transformations complete. Final dataset: {len(train)} rows")
    return train


def upload_biweekly_data(s3_client, bucket_name, processed_data, year, biweek_num):
    """Upload processed parquet and metadata to S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        processed_data: Processed DataFrame
        year: Year for the path
        biweek_num: Biweek number
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        s3_prefix = get_full_biweekly_prefix(year, biweek_num, IO.OUTPUT)
        
        print(f"\n[S3] Uploading processed data to s3://{bucket_name}/{s3_prefix}")
        
        # Upload parquet file
        parquet_buffer = BytesIO()
        processed_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_key = f"{s3_prefix}data.parquet"
        print(f"  Uploading data.parquet...", end=" ")
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=parquet_buffer.getvalue())
        print("✓")
        
        print(f"\n✓ Successfully uploaded to s3://{bucket_name}/{s3_prefix}")
        return True
        
    except Exception as e:
        print(f"✗ Error uploading to S3: {e}")
        return False


def lambda_handler(event, context):
    """AWS Lambda handler for biweekly SARIMAX Prime processing.
    
    Args:
        event: Lambda event from Step Functions with keys:
            - date: ISO date string (YYYY-MM-DD)
            - year: Integer year
            - biweek_num: Integer biweek number
            - biweek_start: ISO date string (YYYY-MM-DD)
            - biweek_end: ISO date string (YYYY-MM-DD)
        context: Lambda context
    
    Returns:
        dict: Response with status and message
    """
    env_name = os.environ.get('ENVIRONMENT', 'dev')
    
    try:
        print("=" * 70)
        print("Biweekly SARIMAX Prime Data Processing Lambda")
        print("=" * 70)
        
        # Get bucket name and input parameters
        s3_client = boto3.client('s3')
        try:
            bucket_name = os.environ.get('DATA_BUCKET')
            if not bucket_name:
                raise ValueError("DATA_BUCKET environment variable not set")
            print(f"Environment: {env_name}")
            print(f"Bucket: {bucket_name}\n")
        except Exception as e:
            error_msg = f"Failed to retrieve bucket name: {e}"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 1: Extract payload parameters from Step Functions
        print("[1/7] Extracting parameters from Step Functions payload...")
        try:
            date_str = event.get('date')
            year = event.get('year')
            biweek_num = event.get('biweek_num')
            biweek_start_str = event.get('biweek_start')
            biweek_end_str = event.get('biweek_end')
            
            if not all([date_str, year is not None, biweek_num is not None, biweek_start_str, biweek_end_str]):
                raise ValueError("Missing required payload parameters: date, year, biweek_num, biweek_start, biweek_end")
            
            # Parse date strings to datetime objects
            process_date = datetime.strptime(date_str, '%Y-%m-%d')
            biweek_start = datetime.strptime(biweek_start_str, '%Y-%m-%d')
            biweek_end = datetime.strptime(biweek_end_str, '%Y-%m-%d')
            
            print(f"✓ Date: {process_date.date()}")
            print(f"✓ Year: {year}, Biweek: {biweek_num}")
            print(f"✓ Period: {biweek_start.date()} to {biweek_end.date()}")
        except (KeyError, ValueError, TypeError) as e:
            error_msg = f"Failed to extract payload parameters: {e}"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 2: Check if processing has already been completed
        print("\n[2/7] Checking if processing has already been completed...")
        if marker_exists(s3_client, bucket_name, get_full_biweekly_prefix(year, biweek_num, IO.OUTPUT)):
            print("✗ Error: Processing workflow has already been completed.")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Processing workflow has already been completed.'})
            }
        print("✓ No marker found. Ready to proceed with processing.")
        
        # Step 3: Load and concatenate daily CSVs
        print("\n[3/7] Loading daily CSVs from biweek period...")
        datasets = load_daily_csvs(s3_client, bucket_name, biweek_start, biweek_end)
        if datasets is None:
            error_msg = "Failed to load datasets from daily CSVs"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 7: Load stores CSV
        print("\n[4/7] Loading stores.csv for feature engineering...")
        stores = load_stores_csv(s3_client, bucket_name)
        if stores is None:
            error_msg = "Failed to load stores.csv"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        datasets['stores'] = stores
        
        # Step 8: Apply SARIMAX Prime transformations
        print("\n[5/7] Processing data with SARIMAX Prime transformations...")
        processed_data = apply_sarimax_prime_transforms(datasets)
        
        # Step 9: Upload to S3
        print("\n[6/7] Uploading processed data to S3...")
        upload_success = upload_biweekly_data(s3_client, bucket_name, processed_data, year, biweek_num)
        
        if not upload_success:
            error_msg = "Failed to upload processed data to S3"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 10: Write marker
        print("\n[7/7] Finalizing...")
        write_marker(s3_client, bucket_name, get_full_biweekly_prefix(year, biweek_num, IO.OUTPUT))
        
        print("\n" + "=" * 70)
        print("✓ Processing completed successfully!")
        print("=" * 70)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Biweekly processing completed',
                'year': year,
                'biweek': biweek_num,
                'period_start': biweek_start.date().isoformat(),
                'period_end': biweek_end.date().isoformat(),
                'rows_processed': len(processed_data)
            })
        }
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }
