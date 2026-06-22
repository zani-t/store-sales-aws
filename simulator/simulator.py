#!/usr/bin/env python3
"""
Data simulator: Query the next n days of data and upload to S3.

Usage:
    python simulator.py <n>

where <n> is the number of days to simulate.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import duckdb
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Add parent directory to path to import bootstrap
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bootstrap import get_stack_output

# Configuration
CUTOFF_DATE = pd.Timestamp('2017-07-01')
DATA_DIR = './simulator/store-sales-time-series-forecasting/'
ENV_NAME = os.environ.get('ENV_NAME', 'dev')


def query_data_by_date(query_date: str):
    """
    Query data from four CSV files for a given date using DuckDB.
    
    Args:
        query_date: Date to query (str in 'YYYY-MM-DD' format)
    
    Returns:
        Tuple of (holidays_df, oil_df, train_df, transactions_df)
    
    Raises:
        ValueError: If train.csv has no data for the given date
    """
    if isinstance(query_date, pd.Timestamp):
        query_date = query_date.strftime('%Y-%m-%d')
    elif isinstance(query_date, datetime):
        query_date = query_date.strftime('%Y-%m-%d')
    
    conn = duckdb.connect(':memory:')
    
    # Query holidays_events.csv
    holidays_result = conn.execute(f"""
        SELECT date, type, locale, locale_name, description, transferred
        FROM read_csv('{DATA_DIR}holidays_events.csv')
        WHERE date = '{query_date}'
    """).fetchall()
    
    if holidays_result:
        holidays_df = pd.DataFrame(
            holidays_result,
            columns=['date', 'type', 'locale', 'locale_name', 'description', 'transferred']
        )
    else:
        holidays_df = pd.DataFrame({
            'date': [query_date],
            'type': [None],
            'locale': [None],
            'locale_name': [None],
            'description': [None],
            'transferred': [None]
        })
    
    # Query oil.csv
    oil_result = conn.execute(f"""
        SELECT date, dcoilwtico
        FROM read_csv('{DATA_DIR}oil.csv')
        WHERE date = '{query_date}'
    """).fetchall()
    
    if oil_result:
        oil_df = pd.DataFrame(oil_result, columns=['date', 'dcoilwtico'])
    else:
        oil_df = pd.DataFrame({
            'date': [query_date],
            'dcoilwtico': [None]
        })
    
    # Query train.csv (must have data or raise error)
    train_result = conn.execute(f"""
        SELECT id, date, store_nbr, family, sales, onpromotion
        FROM read_csv('{DATA_DIR}train.csv')
        WHERE date = '{query_date}'
    """).fetchall()
    
    if not train_result:
        raise ValueError(f"No data found in train.csv for date {query_date}")
    
    train_df = pd.DataFrame(
        train_result,
        columns=['id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion']
    )
    
    # Query transactions.csv
    transactions_result = conn.execute(f"""
        SELECT date, store_nbr, transactions
        FROM read_csv('{DATA_DIR}transactions.csv')
        WHERE date = '{query_date}'
    """).fetchall()
    
    if transactions_result:
        transactions_df = pd.DataFrame(
            transactions_result,
            columns=['date', 'store_nbr', 'transactions']
        )
    else:
        transactions_df = pd.DataFrame({
            'date': [query_date],
            'store_nbr': [None],
            'transactions': [None]
        })
    
    conn.close()
    
    return holidays_df, oil_df, train_df, transactions_df


def find_latest_date_in_s3(s3_client, bucket_name):
    """
    Scan S3 prefix 'raw/daily' for the latest date of uploaded data.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
    
    Returns:
        pd.Timestamp of the latest date, or None if no data found
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='raw/daily/')
        
        dates = []
        
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                # Parse date from path: raw/daily/YYYY/MM/DD/...
                parts = key.split('/')
                if len(parts) >= 4:
                    try:
                        year = int(parts[2])
                        month = int(parts[3])
                        day = int(parts[4])
                        date_obj = pd.Timestamp(year=year, month=month, day=day)
                        dates.append(date_obj)
                    except (ValueError, IndexError):
                        continue
        
        if dates:
            return pd.Timestamp(max(dates))
        return None
    
    except Exception as e:
        print(f"Error scanning S3: {e}")
        return None


def upload_dataframes_to_s3(s3_client, bucket_name, target_date, 
                            holidays_df, oil_df, train_df, transactions_df):
    """
    Upload dataframes as CSV files to S3 at the path:
    <bucket>/raw/daily/<year>/<month>/<day>/<dataframe>.csv
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        target_date: Date object or string in 'YYYY-MM-DD' format
        holidays_df, oil_df, train_df, transactions_df: DataFrames to upload
    """
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)
    
    year = target_date.year
    month = str(target_date.month).zfill(2)
    day = str(target_date.day).zfill(2)
    prefix = f"raw/daily/{year}/{month}/{day}"
    
    dataframes = {
        'holidays_events': holidays_df,
        'oil': oil_df,
        'train': train_df,
        'transactions': transactions_df
    }
    
    for name, df in dataframes.items():
        key = f"{prefix}/{name}.csv"
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=csv_buffer.getvalue()
            )
            print(f"  ✓ Uploaded {key}")
        except ClientError as e:
            print(f"  ✗ Failed to upload {key}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Simulate n days of data ingestion'
    )
    parser.add_argument('n', type=int, help='Number of days to simulate')
    args = parser.parse_args()
    
    n = args.n
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Get data bucket name
    try:
        bucket_name = get_stack_output(ENV_NAME, f"{ENV_NAME}-DataBucketName")
        print(f"Using data bucket: {bucket_name}")
    except Exception as e:
        print(f"Error: Could not retrieve data bucket name: {e}")
        sys.exit(1)
    
    # Find the latest date in S3
    latest_date_in_s3 = find_latest_date_in_s3(s3_client, bucket_name)
    
    if latest_date_in_s3:
        start_date = latest_date_in_s3 + timedelta(days=1)
        print(f"Latest date in S3: {latest_date_in_s3.strftime('%Y-%m-%d')}")
        print(f"Starting from: {start_date.strftime('%Y-%m-%d')}")
    else:
        start_date = CUTOFF_DATE
        print(f"No data found in S3. Starting from CUTOFF_DATE: {start_date.strftime('%Y-%m-%d')}")
    
    # Simulate n days of data
    print(f"\nSimulating {n} days of data...\n")
    
    current_date = start_date
    for i in range(n):
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"[{i+1}/{n}] Querying data for {date_str}...")
        
        try:
            holidays_df, oil_df, train_df, transactions_df = query_data_by_date(date_str)
            upload_dataframes_to_s3(
                s3_client, 
                bucket_name, 
                current_date,
                holidays_df, 
                oil_df, 
                train_df, 
                transactions_df
            )
            print(f"  ✓ Completed {date_str}")
        
        except ValueError as e:
            print(f"  ✗ Error: {e}")
            print(f"Stopping simulation at day {i+1}/{n}")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            sys.exit(1)
        
        current_date += timedelta(days=1)
    
    print(f"\n✓ Successfully simulated {n} days of data")


if __name__ == '__main__':
    main()
