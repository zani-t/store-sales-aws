#!/usr/bin/env python3
"""
Load files from Kaggle API, split by cutoff date, and upload historical data to S3.
"""

import os
import sys
import subprocess
import shutil
import zipfile
import tempfile
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import boto3
from botocore.exceptions import ClientError

from bootstrap import get_stack_output, write_marker

# Configuration
KAGGLE_COMPETITION = 'store-sales-time-series-forecasting'
CUTOFF_DATE = pd.Timestamp('2017-06-30')
DATASET_NAMES = ['holidays_events', 'oil', 'stores', 'train', 'transactions']

# Create temporary directory for all local operations
TEMP_DIR = Path(tempfile.mkdtemp(prefix='raw_data_'))
EXTRACT_DIR = TEMP_DIR / KAGGLE_COMPETITION
HISTORICAL_DIR = TEMP_DIR / 'historical'
ZIP_PATH = TEMP_DIR / f'{KAGGLE_COMPETITION}.zip'


def download_and_extract():
    """Download dataset from Kaggle and extract it to temporary directory."""
    import subprocess
    
    # Check if kaggle CLI is installed and configured
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=False)
        if result.returncode != 0:
            raise FileNotFoundError("Kaggle CLI not found")
    except FileNotFoundError:
        print("Error: Kaggle CLI is not installed or not in PATH.")
        print("Install it with: pip install kaggle")
        print("Configure credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)
    
    # Change to temporary directory for download
    original_cwd = os.getcwd()
    os.chdir(TEMP_DIR)
    
    try:
        # Remove old extracted directory and zip if they exist
        if EXTRACT_DIR.exists():
            print(f"Removing existing extracted directory: {EXTRACT_DIR}")
            shutil.rmtree(EXTRACT_DIR)
        
        if ZIP_PATH.exists():
            print(f"Removing existing zip file: {ZIP_PATH}")
            ZIP_PATH.unlink()
        
        # Download from Kaggle
        print(f"Downloading {KAGGLE_COMPETITION} from Kaggle...")
        try:
            subprocess.run(['kaggle', 'competitions', 'download', '-c', KAGGLE_COMPETITION],
                          check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading from Kaggle: {e}")
            sys.exit(1)
        
        if not ZIP_PATH.exists():
            print(f"Error: Download may have failed. {ZIP_PATH} not found.")
            sys.exit(1)
        
        # Extract zip file
        print(f"Extracting {ZIP_PATH}...")
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            print(f"Successfully extracted to {EXTRACT_DIR}")
        except zipfile.BadZipFile as e:
            print(f"Error: Invalid zip file {ZIP_PATH}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error extracting zip file: {e}")
            sys.exit(1)
        
        # Clean up zip file
        print(f"Removing zip file: {ZIP_PATH}")
        ZIP_PATH.unlink()
    
    finally:
        os.chdir(original_cwd)


def create_directories():
    """Create output directories for historical data."""
    if HISTORICAL_DIR.exists():
        print(f"Removing existing directory: {HISTORICAL_DIR}")
        shutil.rmtree(HISTORICAL_DIR)
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {HISTORICAL_DIR}")


def load_and_split_data():
    """Load CSV files and split by cutoff date."""
    print(f"\nLoading datasets with cutoff date: {CUTOFF_DATE}")
    
    # Load datasets
    input_datasets = {}
    missing_files = []
    
    for dataset_name in DATASET_NAMES:
        csv_path = EXTRACT_DIR / f'{dataset_name}.csv'
        
        if not csv_path.exists():
            missing_files.append(csv_path)
            print(f"Warning: {csv_path} not found")
            continue
        
        try:
            print(f"Loading {dataset_name}...")
            if dataset_name == 'stores':
                input_datasets[dataset_name] = pd.read_csv(csv_path)
            else:
                input_datasets[dataset_name] = pd.read_csv(csv_path, parse_dates=['date'])
            print(f"  ✓ {dataset_name}: {len(input_datasets[dataset_name])} rows")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            sys.exit(1)
    
    if missing_files:
        print(f"\nWarning: Could not find {len(missing_files)} dataset(s):")
        for path in missing_files:
            print(f"  - {path}")
        if not input_datasets:
            print("Error: No datasets could be loaded")
            sys.exit(1)
    
    # Split by cutoff date
    print(f"\nSplitting datasets by cutoff date ({CUTOFF_DATE})...")
    historical_datasets = {}
    
    for dataset_name, df in input_datasets.items():
        if dataset_name == 'stores':
            historical_datasets[dataset_name] = df
        else:
            historical_datasets[dataset_name] = df[df['date'] <= CUTOFF_DATE]

        hist_count = len(historical_datasets[dataset_name])
        print(f"  {dataset_name}: {hist_count} historical")
    
    # Save split datasets
    print(f"\nSaving datasets to output directory...")
    
    for dataset_name in input_datasets.keys():
        try:
            hist_path = HISTORICAL_DIR / f'{dataset_name}.csv'
            historical_datasets[dataset_name].to_csv(hist_path, index=False)
            
            print(f"  ✓ {dataset_name} saved")
        except Exception as e:
            print(f"Error saving {dataset_name}: {e}")
            sys.exit(1)
    
    print("\n✓ All datasets successfully loaded and split!")


def cleanup_temp_files():
    """Remove temporary directory after processing."""
    if TEMP_DIR.exists():
        print(f"\n[CLEANUP] Removing temporary directory: {TEMP_DIR}")
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✓ Cleaned up temporary files")
        except Exception as e:
            print(f"✗ Error removing {TEMP_DIR}: {e}")
            return False
    return True


def upload_to_s3(s3_bucket_name):
    """Upload historical data to S3 bucket."""
    if not s3_bucket_name:
        print("\nWarning: S3 bucket name not provided. Skipping S3 upload.")
        return False
    
    print(f"\n[S3] Connecting to bucket: {s3_bucket_name}")
    
    try:
        s3_client = boto3.client('s3')
        
        # Verify bucket exists and is accessible
        try:
            s3_client.head_bucket(Bucket=s3_bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"Error: Bucket {s3_bucket_name} not found.")
            elif error_code == '403':
                print(f"Error: Access denied to bucket {s3_bucket_name}.")
            else:
                print(f"Error accessing bucket {s3_bucket_name}: {error_code}")
            return False
        
        print(f"✓ Bucket accessible")
        
        # Upload historical data
        print(f"\n[S3] Uploading historical data to s3://{s3_bucket_name}/raw/historical/")
        
        if not HISTORICAL_DIR.exists():
            print(f"Error: Historical directory not found: {HISTORICAL_DIR}")
            return False
        
        uploaded_count = 0
        for csv_file in HISTORICAL_DIR.glob('*.csv'):
            s3_key = f"raw/historical/{csv_file.name}"
            try:
                print(f"  Uploading {csv_file.name}...", end=" ")
                s3_client.upload_file(
                    str(csv_file),
                    s3_bucket_name,
                    s3_key
                )
                print(f"✓")
                uploaded_count += 1
            except ClientError as e:
                print(f"✗")
                print(f"  Error uploading {csv_file.name}: {e}")
                return False
        
        print(f"\n✓ Successfully uploaded {uploaded_count} file(s) to S3")
        print(f"  Bucket: {s3_bucket_name}")
        print(f"  Path: s3://{s3_bucket_name}/raw/historical/")
        
        return True
        
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return False


def main(env_name):
    """Main entry point.
    
    Args:
        env_name: AWS CDK environment name (e.g., 'dev', 'prod')
    """
    # Get S3 bucket name from CDK CloudFormation exports
    s3_bucket_name = get_stack_output(env_name, f"{env_name}-DataBucketName")
    
    print("=" * 60)
    print("Store Sales Time Series Forecasting Dataset Loader")
    print("=" * 60)
    print(f"Environment: {env_name}")
    if s3_bucket_name:
        print(f"S3 Bucket: {s3_bucket_name}")
    print()
    
    try:
        # Download and extract
        print("[1/5] Downloading and extracting dataset...")
        download_and_extract()
        
        # Create output directories
        print("\n[2/5] Creating output directories...")
        create_directories()
        
        # Load and split data
        print("\n[3/5] Loading and splitting data...")
        load_and_split_data()
        
        # Upload to S3
        print("\n[4/5] Uploading to S3...")
        upload_success = upload_to_s3(s3_bucket_name)
        
        if upload_success:
            write_marker(s3_bucket_name, "raw/historical/")
        else:
            print("Upload failed. Aborting.")
            cleanup_temp_files()
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("✓ Process completed successfully!")
        print("=" * 60)
        
        # Clean up temporary directory
        cleanup_temp_files()
        
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
        print("Usage: python 1_raw.py <env_name>")
        print("Example: python 1_raw.py dev")
        sys.exit(1)
    
    env_name = sys.argv[1]
    main(env_name)

