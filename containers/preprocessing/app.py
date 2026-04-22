import os
import json
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
BIWEEKLY_DATASET_NAMES = ['holidays_events', 'oil', 'train', 'transactions']
BIWEEKLY_OUTPUT_PREFIX = 'processed/sarimax-prime/biweekly/'
CUTOFF_DATE = datetime(2017, 7, 15)  # Storage starts at 2017/BW-14


def get_full_output_prefix(year, biweek_num):
    return f"{BIWEEKLY_OUTPUT_PREFIX}{year}/BW-{biweek_num}/"


def marker_exists(s3_client, bucket, prefix):
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}{MARKER}")
        return True
    except ClientError:
        return False


def write_marker(s3_client, bucket, prefix):
    s3_client.put_object(Bucket=bucket, Key=f"{prefix}{MARKER}", Body=b'')


def get_latest_daily_folder(s3_client, bucket_name):
    """Get the latest folder date from raw/daily/.
    
    Returns:
        tuple: (datetime object, folder_prefix) or (None, None) if no folders found
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='raw/daily/')
        
        latest_date = None
        latest_prefix = None
        
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                # Parse folders like raw/daily/2024/01/15/
                parts = key.rstrip('/').split('/')
                if len(parts) >= 5 and parts[0] == 'raw' and parts[1] == 'daily' and parts[2] and parts[3] and parts[4]:
                    try:
                        year, month, day = int(parts[2]), int(parts[3]), int(parts[4])
                        folder_date = datetime(year, month, day)
                        if latest_date is None or folder_date > latest_date:
                            latest_date = folder_date
                            latest_prefix = f"raw/daily/{year}/{month:02d}/{day:02d}/"
                    except (ValueError, TypeError):
                        continue
        
        return latest_date, latest_prefix
    except Exception as e:
        print(f"Error listing daily folders: {e}")
        return None, None


def is_trigger_date(date_obj):
    """Check if date is a trigger date."""
    last_day = monthrange(date_obj.year, date_obj.month)[1]
    return date_obj.day == 15 or date_obj.day == last_day


def calculate_biweek_number(date_obj):
    """Calculate biweek number starting from CUTOFF_DATE (2017/07/15 = BW-13).
    
    Returns:
        tuple: (year, biweek_number, start_date, end_date)
    """
    # Extract year and month to determine biweek label
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    
    # Determine which biweek within the month
    if day <= 15:
        biweek_in_month = 1
        biweek_start = datetime(year, month, 1)
        biweek_end = datetime(year, month, 15)
    else:
        biweek_in_month = 2
        biweek_start = datetime(year, month, 16)
        # End of month
        if month == 12:
            biweek_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            biweek_end = datetime(year, month + 1, 1) - timedelta(days=1)

    # Calculate biweeks since cutoff
    months_since_cutoff = (date_obj.year - CUTOFF_DATE.year) * 12 + (date_obj.month - CUTOFF_DATE.month)
    biweeks_since_cutoff = (months_since_cutoff * 2) + (biweek_in_month - 1)
    
    # BW-14 starts at CUTOFF_DATE
    global_biweek = 13 + biweeks_since_cutoff
    
    return year, global_biweek, biweek_start, biweek_end


def load_lambda_hmv_jsons(s3_client, bucket_name, year, biweek_num):
    """Load lambda and HMV JSON files from S3.
    
    Returns:
        tuple: (lambdas dict, hmvs dict) or (None, None) if error
    """
    try:
        print(f"\n[S3] Loading lambda and HMV values...")
        if biweek_num == 14:
            json_prefix = 'processed/sarimax-prime/historical/'
        else:
            json_prefix = get_full_output_prefix(year, biweek_num - 1)
        
        # Load lambda values
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=f"{json_prefix}lambdas.json")
            lambdas = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded lambdas: {lambdas}")
        except Exception as e:
            print(f"  ✗ Could not load lambdas.json: {e}")
            return None, None
        
        # Load HMV values
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=f"{json_prefix}hmvs.json")
            hmvs = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded HMV values for {len(hmvs)} holidays")
        except Exception as e:
            print(f"  ✗ Could not load hmvs.json: {e}")
            return None, None
        
        return lambdas, hmvs
    
    except Exception as e:
        print(f"Error loading lambda/HMV JSONs: {e}")
        return None, None


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


def apply_sarimax_prime_transforms(datasets, lambdas, hmvs):
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
    
    # BoxCox transforms
    print("  - Applying BoxCox transforms...")
    train['onpromotion'] = boxcox(train['onpromotion'] + 0.01, lambdas['lmbda_onpromotion'])
    train['transactions'] = boxcox(train['transactions'] + 0.01, lambdas['lmbda_transactions'])
    train['sales'] = boxcox(train['sales'] + 0.01, lambdas['lmbda_sales'])

    # Target encoding - HolidayMeanVariation
    print("  - Computing holiday mean variations...")
    ma = train[['date', 'sales']].groupby(['date']).agg({'sales': 'mean'})
    ma = pd.DataFrame(ma.rolling(window=13, min_periods=13).mean().values, columns=['ma30']).set_index(ma.index)
    train = train.merge(ma, how='left', on='date')
    train['hmv'] = 0.0
    for holiday in holidays_events['description'].unique():
        if holiday not in hmvs:
            df = train.loc[train['description'] == holiday, ['date', 'ma30', 'sales']].groupby(['date', 'ma30'], as_index=False).agg(sales=('sales', 'mean'))
            hmv = (df['sales'] - df['ma30']).mean()
            hmvs[holiday] = float(hmv)
        else:
            hmv = hmvs[holiday]
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
    
    train = train.reindex(columns=train.columns.union(cols_to_int), fill_value=0)
    train[cols_to_int] = train[cols_to_int].astype('int8')
    
    train = train.drop(['locale', 'locale_name', 'description', 'transferred', 'ma30'], axis=1)
    
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
        s3_prefix = get_full_output_prefix(year, biweek_num)
        
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
        event: Lambda event (unused for S3 trigger)
        context: Lambda context
    
    Returns:
        dict: Response with status and message
    """
    env_name = os.environ.get('ENVIRONMENT', 'dev')
    
    try:
        print("=" * 70)
        print("Biweekly SARIMAX Prime Data Processing Lambda")
        print("=" * 70)
        
        # Get bucket name from CloudFormation exports
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
        
        # Step 1: Get latest daily folder
        print("[1/10] Checking latest daily upload...")
        latest_date, latest_prefix = get_latest_daily_folder(s3_client, bucket_name)
        
        if latest_date is None:
            error_msg = "No daily folders found in raw/daily/"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        print(f"✓ Latest folder: {latest_date.date()}")
        
        # Step 2: Check if trigger date
        print("\n[2/10] Checking if date is a trigger date...")
        if not is_trigger_date(latest_date):
            msg = f"Date {latest_date.date()} is not a trigger date. Skipping processing."
            print(f"ℹ {msg}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': msg})
            }
        print(f"✓ {latest_date.date()} is a trigger date")
        
        # Step 3: Calculate biweek
        print("\n[3/10] Calculating biweek parameters...")
        year, biweek_num, biweek_start, biweek_end = calculate_biweek_number(latest_date)
        print(f"✓ Year: {year}, Biweek: {biweek_num}")
        print(f"  Period: {biweek_start.date()} to {biweek_end.date()}")
        
        # Step 4: Check if processing has already been completed
        print("\n[4/10] Checking if processing has already been completed...")
        if marker_exists(s3_client, bucket_name, get_full_output_prefix(year, biweek_num)):
            print("✗ Error: Processing workflow has already been completed.")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Processing workflow has already been completed.'})
            }
        print("✓ No marker found. Ready to proceed with processing.")
        
        # Step 5: Load previous lambda and HMV jsons
        print("\n[5/10] Loading previous lambda and HMV values...")
        lambdas, hmvs = load_lambda_hmv_jsons(s3_client, bucket_name, year, biweek_num)
        if lambdas is None or hmvs is None:
            error_msg = "Failed to load lambda or HMV values"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 6: Load and concatenate daily CSVs
        print("\n[6/10] Loading daily CSVs from biweek period...")
        datasets = load_daily_csvs(s3_client, bucket_name, biweek_start, biweek_end)
        if datasets is None:
            error_msg = "Failed to load datasets from daily CSVs"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 7: Load stores CSV
        print("\n[7/10] Loading stores.csv for feature engineering...")
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
        print("\n[8/10] Processing data with SARIMAX Prime transformations...")
        processed_data = apply_sarimax_prime_transforms(datasets, lambdas, hmvs)
        
        # Step 9: Upload to S3
        print("\n[9/10] Uploading processed data to S3...")
        upload_success = upload_biweekly_data(s3_client, bucket_name, processed_data, year, biweek_num)
        
        if not upload_success:
            error_msg = "Failed to upload processed data to S3"
            print(f"✗ {error_msg}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': error_msg})
            }
        
        # Step 10: Write marker
        print("\n[10/10] Finalizing...")
        write_marker(s3_client, bucket_name, get_full_output_prefix(year, biweek_num))
        
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
