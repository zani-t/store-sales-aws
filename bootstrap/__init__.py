from io import BytesIO

import boto3
from botocore.exceptions import ClientError

MARKER = '_COMPLETE'
SIGNIFICANT_EXOG = ['hmv', 'exists_promotion', 'exists_transaction']

s3 = boto3.client('s3')


def get_stack_output(env_name, export_name):
    cf = boto3.client('cloudformation')
    response = cf.describe_stacks(StackName=f"{env_name}-StorageStack")
    outputs = response['Stacks'][0]['Outputs']
    return next(o['OutputValue'] for o in outputs if o['ExportName'] == export_name)


def marker_exists(bucket, prefix):
    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix}{MARKER}")
        return True
    except ClientError:
        return False


def write_marker(bucket, prefix):
    s3.put_object(Bucket=bucket, Key=f"{prefix}{MARKER}", Body=b'')


def family_encode(name):
    return name.replace(' ', '_').replace('/', '_')


def load_time_series(s3_client, bucket_name):
    """Load pre-built time series aggregations from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
    
    Returns:
        tuple: (ts_per_family dict, ts_per_store dict)
    """
    import json
    import pandas as pd

    print(f"\n[TIMESERIES] Loading time series from S3")
    
    ts_per_family = {}
    ts_per_store = {}
    
    try:
        # Load family name mapping from S3
        print(f"  Loading family name mapping...")
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key='processed/sarimax-prime/historical/families_mapping.json')
            family_mapping = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"    Warning: Could not load family mapping: {e}. Aborting.")
            return None, None

        # Load family time series
        print(f"  Loading family time series...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix='processed/sarimax-prime/historical/family/')
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.parquet'):
                    try:
                        response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                        ts_data = pd.read_parquet(BytesIO(response['Body'].read()))
                        # Extract encoded family name from filename and use mapping
                        encoded_name = obj['Key'].split('/')[-1].replace('.parquet', '')
                        try:
                            family_name = family_mapping[encoded_name]
                        except KeyError:
                            print(f"  ✗ Error: Encoded family name '{encoded_name}' not found in mapping. Aborting.")
                            return None, None

                        ts_per_family[family_name] = ts_data
                    except Exception as e:
                        print(f"  ✗ Could not load family time series from {obj['Key']}: {e}")
                        return None, None
        
        print(f"    ✓ Loaded {len(ts_per_family)} family time series")
        
        # Load store time series
        print(f"  Loading store time series...")
        pages = paginator.paginate(Bucket=bucket_name, Prefix='processed/sarimax-prime/historical/store/')
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.parquet'):
                    try:
                        response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                        ts_data = pd.read_parquet(BytesIO(response['Body'].read()))
                        # Extract store number from filename (e.g., "store_01.parquet" -> 1)
                        filename = obj['Key'].split('/')[-1].replace('.parquet', '')
                        if filename.startswith('store_'):
                            store_num = int(filename.split('_')[1])
                            ts_per_store[store_num] = ts_data
                        else:
                            print(f"  ✗ Warning: Unexpected store time series filename '{filename}'. Aborting.")
                            return None, None
                        
                    except Exception as e:
                        print(f"    Warning: Could not load store time series from {obj['Key']}: {e}")
                        return None, None
        
        print(f"    ✓ Loaded {len(ts_per_store)} store time series")
        
        return ts_per_family, ts_per_store
        
    except Exception as e:
        print(f"  ✗ Error loading time series: {e}")
        return None, None
