import os
import json
import pickle
import joblib
import datetime
import uuid
from enum import Enum
from pathlib import Path
from io import BytesIO
from datetime import datetime as dt
from calendar import monthrange
from decimal import Decimal

import boto3
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from botocore.exceptions import ClientError
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


MARKER = '_COMPLETE'

SUBPRIME_INPUT_PREFIX = 'processed/sarimax-subprime/biweekly/'
PRIME_INPUT_PREFIX = 'processed/sarimax-prime/biweekly/'
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


def marker_exists(s3_client, bucket, prefix):
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}{MARKER}")
        return True
    except ClientError:
        return False


def get_latest_biweek(s3_client, bucket):
    """Determine the latest biweek number available in S3 based on folder structure.
    
    Args:
        s3_client: The S3 client.
        bucket: The S3 bucket name.

    Returns:
        A tuple containing the latest year and biweek number.
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        latest_year = None
        latest_biweek = None
        
        # First level: get year folders
        pages = paginator.paginate(Bucket=bucket, Prefix=SUBPRIME_INPUT_PREFIX, Delimiter='/')
        year_folders = []
        for page in pages:
            for prefix in page.get('CommonPrefixes', []):
                year_folders.append(prefix['Prefix'])
        
        # Second level: for each year, get biweek folders
        for year_folder in year_folders:
            pages = paginator.paginate(Bucket=bucket, Prefix=year_folder, Delimiter='/')
            for page in pages:
                for prefix in page.get('CommonPrefixes', []):
                    biweek_folder = prefix['Prefix']
                    # Extract year and biweek from path like "processed/sarimax-subprime/biweekly/2017/BW-14/"
                    parts = biweek_folder.rstrip('/').split('/')
                    if len(parts) >= 2:
                        try:
                            year = int(parts[-2])
                            biweek_str = parts[-1]
                            if biweek_str.startswith('BW-'):
                                biweek_num = int(biweek_str.replace('BW-', ''))
                                if (latest_biweek is None) or (latest_year is None) or (year > latest_year) or (year == latest_year and biweek_num > latest_biweek):
                                    latest_year = year
                                    latest_biweek = biweek_num
                        except (ValueError, IndexError):
                            continue
        
        return latest_year, latest_biweek
    except ClientError as e:
        print(f"Error fetching latest biweek number: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error fetching latest biweek number: {e}")
        return None, None


def load_subprime_data(s3_client, bucket_name, year, biweek_num):
    """Load subprime data from S3 for the specified year and biweek number.
    
    Args:
        s3_client: The S3 client.
        bucket_name: The S3 bucket name.
        year: The year for which to load data.
        biweek_num: The biweek number for which to load data.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=f'{SUBPRIME_INPUT_PREFIX}{year}/BW-{biweek_num}/data.parquet')
        data = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"✓ Successfully downloaded data from s3://{bucket_name}/{SUBPRIME_INPUT_PREFIX}{year}/BW-{biweek_num}/data.parquet")
        return data
    except ClientError as e:
        error_msg = f"Failed to load data from S3: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)


def load_jsons(s3_client, bucket_name, year, biweek_num):
    """Load lambda and HMV values from S3 for the specified year and biweek number.
    
    Args:
        s3_client: The S3 client.
        bucket_name: The S3 bucket name.
        year: The year for which to load data.
        biweek_num: The biweek number for which to load data.

    Returns:
        A tuple containing the loaded lambda values, HMV values, and families mapping.
    """
    try:
        print(f"\n[S3] Loading lambda and HMV values...")
        if biweek_num == 14:
            json_prefix = 'processed/sarimax-prime/historical/'
        else:
            json_prefix = f"{PRIME_INPUT_PREFIX}{year}/BW-{biweek_num - 1}/"
        
        # Load lambda values
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=f"{json_prefix}lambdas.json")
            lambdas = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded lambdas: {lambdas}")
        except Exception as e:
            print(f"  ✗ Could not load lambdas.json: {e}")
            return None, None, None
        
        # Load HMV values
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=f"{json_prefix}hmvs.json")
            hmvs = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded HMV values for {len(hmvs)} holidays")
        except Exception as e:
            print(f"  ✗ Could not load hmvs.json: {e}")
            return None, None, None
        
        # Load families mapping
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key="processed/sarimax-prime/historical/families_mapping.json")
            families = json.loads(response['Body'].read().decode('utf-8'))
            print(f"  ✓ Loaded families mapping for {len(families)} families")
        except Exception as e:
            print(f"  ✗ Could not load families_mapping.json: {e}")
            return None, None, None
        
        return lambdas, hmvs, families
    except ClientError as e:
        error_msg = f"Failed to load previous biweek jsons from S3: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)
    

def prime_data_for_sarimax(data, lambdas, hmvs):
    """Apply Box-Cox transformations and fill HMV values in preparation for SARIMAX modeling.
    
    Args:
        data: The input DataFrame.
        lambdas: A dictionary containing the lambda values for Box-Cox transformations.
        hmvs: A dictionary containing the HMV values for each holiday.

    Returns:
        A pandas DataFrame containing the transformed data.
    """
    try:
        print("\n[Data Preparation] Applying remaining transformations for SARIMAX...")

        # Apply Box-Cox transformations
        data['onpromotion'] = boxcox(data['onpromotion'] + 0.01, lambdas['lmbda_onpromotion'])
        data['transactions'] = boxcox(data['transactions'] + 0.01, lambdas['lmbda_transactions'])
        data['sales'] = boxcox(data['sales'] + 0.01, lambdas['lmbda_sales'])

        # Fill HMVs
        ma = data[['date', 'sales']].groupby(['date']).agg({'sales': 'mean'})
        ma = pd.DataFrame(ma.rolling(window=15, min_periods=15).mean().values, columns=['ma30']).set_index(ma.index)
        data = data.merge(ma, how='left', on='date')
        data['hmv'] = 0.0
        for holiday in data['description'].unique():
            df = data.loc[data['description'] == holiday, ['date', 'ma30', 'sales']].groupby(['date', 'ma30'], as_index=False).agg(sales=('sales', 'mean'))
            hmv = hmvs.get(holiday, float((df['sales'] - df['ma30']).mean()))
            data.loc[data['description'] == holiday, 'hmv'] = ((data['ntl_holiday'] == 1) |
                                                                (data['rgnl_holiday'] == 1) |
                                                                (data['lcl_holiday'] == 1)).astype('int8') * hmv
            
        data = data.drop(['description', 'ma30'], axis=1)
        
        print("✓ Data preparation completed.")
        return data
    except Exception as e:
        error_msg = f"Error during data preparation: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)


def build_time_series(data):
    """Aggregate data to build time series for SARIMAX modeling.
    
    Args:
            data: The input DataFrame.
    Returns:
            A tuple containing two dictionaries: (ts_per_family, ts_per_store)
    """
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
        print(f"  ✗ Error building time series: {e}")
        return None, None
    

def load_sarimax_models(s3_client, bucket_name, families, year, biweek_num):
    """Load previously trained SARIMAX models from S3 for the specified year and biweek number.
    
    Args:
        s3_client: The S3 client.
        bucket_name: The S3 bucket name.
        families: The mapping of model identifiers to family names.
        year: The year for which to load data.
        biweek_num: The biweek number for which to load data.

    Returns:
        A tuple containing the loaded SARIMAX models for families and stores.
    """
    smx_per_family = {}
    smx_per_store = {}

    try:
        if biweek_num == 14:
            model_prefix = 'sarimax/historical/'
        else:
            model_prefix = f"sarimax/biweekly/{year}/BW-{biweek_num - 1}/"
        
        print(f"\n[S3] Downloading SARIMAX models from s3://{bucket_name}/{model_prefix}...")
        # List and download family models
        print(f"  Downloading family models...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{model_prefix}family/")
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.pkl'):
                    model_name = Path(obj['Key']).stem
                    try:
                        response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                    except ClientError as e:
                        print(f"  ✗ Failed to download {obj['Key']}: {e}")
                        continue
                    data_bytes = BytesIO(response['Body'].read())
                    if data_bytes:
                        smx_per_family[families[model_name]] = pickle.load(data_bytes)
        
        print(f"    ✓ Downloaded {len(smx_per_family)} family models")
        
        # List and download store models
        print(f"  Downloading store models...")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{model_prefix}store/")
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                if obj['Key'].endswith('.pkl'):
                    # Extract store number from filename (e.g., "store_01.pkl" -> 1)
                    model_name = Path(obj['Key']).stem
                    if model_name.startswith('store_'):
                        store_num = int(model_name.split('_')[1])
                        try:
                            response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                        except ClientError as e:
                            print(f"  ✗ Failed to download {obj['Key']}: {e}")
                            continue
                        data_bytes = BytesIO(response['Body'].read())
                        if data_bytes:
                            smx_per_store[store_num] = pickle.load(data_bytes)
        
        print(f"    ✓ Downloaded {len(smx_per_store)} store models")
        
        return smx_per_family, smx_per_store
    except ClientError as e:
        error_msg = f"Failed to load previous biweek SARIMAX models from S3: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)
    

def generate_inferences(ts_per_family, ts_per_store, smx_per_family, smx_per_store):
    """Generate predictions from loaded SARIMAX models using time series data.
    
    Args:
        ts_per_family: Dictionary of aggregated family time series
        ts_per_store: Dictionary of aggregated store time series
        smx_per_family: Dictionary of trained family models
        smx_per_store: Dictionary of trained store models
    
    Returns:
        tuple: (forecast_per_family dict, forecast_per_store dict)
    
    Raises:
        Exception: If inference generation fails
    """
    def clean(forecast):
        forecast.rename(columns={0 : 'date'}, inplace=True)
        forecast = forecast.set_index('date').drop('index', axis=1)
        return forecast

    print(f"\n[INFERENCE] Generating SARIMAX predictions")
    
    forecast_per_family = {}
    forecast_per_store = {}
    
    # Generate family predictions
    dates = pd.Series(ts_per_store[list(ts_per_store.keys())[0]].index)

    print(f"  Generating predictions per family...")
    for f in smx_per_family:
        try:
            exog_data = ts_per_family[f][SIGNIFICANT_EXOG]
            if f == 'BOOKS':
                exog_data = exog_data.drop('exists_promotion', axis=1)
            forecast_per_family[f] = clean(
                pd.concat([
                    smx_per_family[f].get_forecast(steps=len(exog_data), exog=exog_data).predicted_mean.reset_index(),
                    dates
                ], axis=1))
        except Exception as e:
            raise Exception(f"Failed to generate inference for family '{f}': {e}")
    
    print(f"    ✓ Generated predictions for {len(forecast_per_family)} families")
    
    # Generate store predictions
    print(f"  Generating predictions per store...")
    for s in smx_per_store:
        try:
            exog_data = ts_per_store[s][SIGNIFICANT_EXOG]
            if s == 25 or s == 52:
                exog_data = exog_data.drop('exists_transaction', axis=1)
            forecast_per_store[s] = clean(
                pd.concat([
                    smx_per_store[s].get_forecast(steps=len(exog_data), exog=exog_data).predicted_mean.reset_index(),
                    dates
                ], axis=1))
        except Exception as e:
            raise Exception(f"Failed to generate inference for store {s}: {e}")
    
    print(f"    ✓ Generated predictions for {len(forecast_per_store)} stores")
    
    return forecast_per_family, forecast_per_store


def eval_sarimax_inferences(forecast_per_family, forecast_per_store, ts_per_family, ts_per_store):
    """Evaluate SARIMAX predictions against actual sales data.
    
    Args:
        forecast_per_family: Dictionary of family predictions
        forecast_per_store: Dictionary of store predictions
        ts_per_family: Dictionary of family time series data
        ts_per_store: Dictionary of store time series data
    
    Returns:
        dict: Evaluation metrics (e.g., RMSLE) for family and store predictions
    
    Raises:
        Exception: If evaluation fails
    """
    print(f"\n[EVALUATION] Evaluating SARIMAX predictions")
    
    try:
        # Evaluate family predictions
        family_metrics = {}
        for f, pred_df in forecast_per_family.items():
            actual = ts_per_family[f]['sales']
            pred = pred_df['predicted_mean']
            common_index = actual.index.intersection(pred.index)
            if len(common_index) > 0:
                rmsle = np.sqrt(mean_squared_error(actual[common_index], pred[common_index]))
                family_metrics[f] = Decimal(str(rmsle))
        
        # Evaluate store predictions
        store_metrics = {}
        for s, pred_df in forecast_per_store.items():
            actual = ts_per_store[s]['sales']
            pred = pred_df['predicted_mean']
            common_index = actual.index.intersection(pred.index)
            if len(common_index) > 0:
                rmsle = np.sqrt(mean_squared_error(actual[common_index], pred[common_index]))
                store_metrics[str(s)] = Decimal(str(rmsle))
        
        print(f"    ✓ Evaluation completed for {len(family_metrics)} families and {len(store_metrics)} stores")
        
        return {
            'family_metrics': family_metrics,
            'store_metrics': store_metrics
        }
    except Exception as e:
        raise Exception(f"Error during evaluation of SARIMAX predictions: {e}")


def prime_data_for_xgboost(data, inf_per_family, inf_per_store):
    """Create complete feature dataframe with SARIMAX predictions.
    
    Args:
        data: Original DataFrame
        inf_per_family: Dictionary of family predictions
        inf_per_store: Dictionary of store predictions
    
    Returns:
        pd.DataFrame: Feature-rich DataFrame
    """
    print(f"\n[FEATURES] Creating feature-rich dataframe")
    
    try:
        X = data.copy()
        
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

        y = X.pop('sales')
        
        return X, y
        
    except Exception as e:
        print(f"  ✗ Error creating dataframe: {e}")
        return None, None
    

def load_xgboost_model(dynamodb_resource, s3_client, table_name, bucket_name):
    """Load previously trained XGBoost model from S3 using path stored in DynamoDB.
    
    Args:
        dynamodb_resource: The DynamoDB resource.
        s3_client: The S3 client.
        table_name: The name of the DynamoDB table containing model metadata.
        bucket_name: The S3 bucket name where the model is stored.

    Returns:
        The loaded XGBoost model.
    """
    try:
        # get S3 path from DynamoDB
        model_item = dynamodb_resource.meta.client.get_item(
            TableName=table_name,
            Key={
                'model': 'xgbsr'
            }
        ).get('Item', {})
        if not model_item:
            raise Exception("Model not found in DynamoDB")
        # extract key from model_path URI and load model from S3
        key = model_item['path'].replace(f"s3://{bucket_name}/", "")
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        model_data = response['Body'].read()
        model_dict = joblib.load(BytesIO(model_data))
        model = model_dict['model']
        feature_names = model_dict['feature_names']
        model_job_id = model_item.get('model_job_id', None)
        print(f"✓ Successfully loaded XGBoost model from s3://{bucket_name}/{key}")
        return model, feature_names, model_job_id
    except ClientError as e:
        error_msg = f"Failed to load latest XGBoost model: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)


def evaluate_xgboost_model(model, X, y, feature_names, lmbda_sales):
    """Evaluate XGBoost model predictions against actual sales data.
    
    Args:
        model: The loaded XGBoost model.
        X: The feature DataFrame.
        y: The target variable (actual sales).
        feature_names: List of feature column names.
        lmbda_sales: The lambda parameter for the Box-Cox transformation.

    Returns:
        rmsle: The evaluation metric (e.g., RMSLE) for the XGBoost model.
    """
    try:
        # add missing feature columns with default value of 0
        for feature in feature_names:
            if feature not in X.columns:
                X[feature] = 0
        X = X[feature_names]
        # generate predictions
        predictions = model.predict(X)
        # inverse boxcox transformation
        predictions = inv_boxcox(predictions, lmbda_sales)
        y = inv_boxcox(y, lmbda_sales)
        # calculate rmsle
        rmsle = np.sqrt(mean_squared_log_error(y, predictions))
        print(f"✓ Evaluation completed. RMSLE: {rmsle}")
        return rmsle
    except Exception as e:
        error_msg = f"Error during model evaluation: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)
    

def save_evaluation_results(dynamodb_resource, job_table_name, model_job_id, year, biweek_num, results):
    """Save evaluation results to DynamoDB with metadata.
    
    Args:
        dynamodb_resource: The DynamoDB resource.
        job_table_name: The name of the DynamoDB table to save results to.
        model_job_id: The job ID of the model being evaluated.
        year: The year of the evaluation.
        biweek_num: The biweek number of the evaluation.
        results: The evaluation results to save.
    
    Returns:
        None
    """
    job_id = str(uuid.uuid4())
    complete_timestamp = str(dt.now(datetime.UTC))[:-6]

    try:
        # save smx evaluation results to DynamoDB
        table = dynamodb_resource.Table(job_table_name)
        item = {
            'job_type': 'eval-sarimax',
            'complete_timestamp': complete_timestamp,
            'job_id': job_id,
            'model_job_id': model_job_id,
            'biweek': f"{year}-BW-{biweek_num}",
            'results': results['sarimax']
        }
        table.put_item(Item=item)
        print(f"✓ Saved evaluation results to DynamoDB.")

        # save XGBoost evaluation results to DynamoDB
        item = {
            'job_type': 'eval-xgboost',
            'complete_timestamp': complete_timestamp,
            'job_id': job_id,
            'model_job_id': model_job_id,
            'biweek': f"{year}-BW-{biweek_num}",
            'results': results['xgbsr']
        }
        table.put_item(Item=item)
        print(f"✓ Saved evaluation results to DynamoDB.")
    except ClientError as e:
        error_msg = f"Failed to save evaluation results to DynamoDB: {e}"
        print(f"✗ {error_msg}")
        raise Exception(error_msg)


if __name__ == "__main__":
    """
    ECS Fargate entry point for biweekly SARIMAX prime data processing and evaluation.
    """
    import sys
    env_name = os.environ.get('ENVIRONMENT', 'dev')
    
    try:
        print("=" * 70)
        print("Biweekly Model Evaluation Process")
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
        except Exception as e:
            error_msg = f"Failed to retrieve bucket name: {e}"
            print(f"✗ {error_msg}")
            sys.exit(1)
        
        # Step 1: Get latest biweek number
        print("[1/13] Getting latest biweek number...")
        year, biweek_num = get_latest_biweek(s3_client, data_bucket_name)
        if year is None or biweek_num is None:
            error_msg = "Could not determine latest biweek number from S3"
            print(f"✗ {error_msg}")
            sys.exit(1)
        print(f"Latest biweek number: {year}-BW-{biweek_num}")

        # Step 2: Check for subprime data marker
        print(f"\n[2/13] Checking for subprime data marker in s3://{data_bucket_name}/{SUBPRIME_INPUT_PREFIX}{year}/BW-{biweek_num}...")
        if not marker_exists(s3_client, data_bucket_name, f"{SUBPRIME_INPUT_PREFIX}{year}/BW-{biweek_num}/"):
            error_msg = f"Subprime data marker not found for {year}-BW-{biweek_num}. Ensure that the data processing step has completed successfully."
            print(f"ℹ {error_msg}")
            sys.exit(0)

        # Step 3: Load subprime data
        print(f"\n[3/13] Loading subprime data for {year}-BW-{biweek_num}...")
        data = load_subprime_data(s3_client, data_bucket_name, year, biweek_num)

        # Step 4: Load previous biweek JSONs (lambdas, HMV values, families mapping)
        print(f"\n[4/13] Loading previous biweek JSONs...")
        lambdas, hmvs, families = load_jsons(s3_client, data_bucket_name, year, biweek_num)
        if lambdas is None or hmvs is None or families is None:
            error_msg = "Failed to load necessary JSON files for data preparation"
            print(f"✗ {error_msg}")
            sys.exit(1)

        # Step 5: Prepare data for SARIMAX
        print(f"\n[5/13] Preparing data for SARIMAX...")
        data = prime_data_for_sarimax(data, lambdas, hmvs)

        # Step 6: Build time series for SARIMAX
        print(f"\n[6/13] Building time series for SARIMAX...")
        ts_per_family, ts_per_store = build_time_series(data)
        if ts_per_family is None or ts_per_store is None:
            error_msg = "Failed to build time series for SARIMAX"
            print(f"✗ {error_msg}")
            sys.exit(1)

        # Step 7: Load SARIMAX models from S3
        print(f"\n[7/13] Loading SARIMAX models from S3...")
        smx_per_family, smx_per_store = load_sarimax_models(s3_client, model_bucket_name, families,year, biweek_num)
        if smx_per_family is None or smx_per_store is None:
            error_msg = "Failed to load SARIMAX models from S3"
            print(f"✗ {error_msg}")
            sys.exit(1)
        
        # Step 8: Generate inferences from SARIMAX models
        print(f"\n[8/13] Generating inferences from SARIMAX models...")
        forecast_per_family, forecast_per_store = generate_inferences(ts_per_family, ts_per_store, smx_per_family, smx_per_store)
        if forecast_per_family is None or forecast_per_store is None:
            error_msg = "Failed to generate inferences from SARIMAX models"
            print(f"✗ {error_msg}")
            sys.exit(1)

        # Step 9: Evaluate SARIMAX inferences
        print(f"\n[9/13] Evaluating SARIMAX inferences...")
        evaluation_results = eval_sarimax_inferences(forecast_per_family, forecast_per_store, ts_per_family, ts_per_store)
        if evaluation_results is None:
            error_msg = "Failed to evaluate SARIMAX inferences"
            print(f"✗ {error_msg}")
            sys.exit(1)

        # Step 10: Prepare data for XGBoost
        print(f"\n[10/13] Preparing data for XGBoost...")
        X, y = prime_data_for_xgboost(data, forecast_per_family, forecast_per_store)
        if X is None or y is None:
            error_msg = "Failed to prepare data for XGBoost"
            print(f"✗ {error_msg}")
            sys.exit(1)

        # Step 11: Load XGBoost model from S3
        print(f"\n[11/13] Loading XGBoost model from S3...")
        model, feature_names, model_job_id = load_xgboost_model(dynamodb_resource, s3_client, model_table_name, model_bucket_name)
        if model is None or feature_names is None or model_job_id is None:
            error_msg = "Failed to load XGBoost model from S3 or model job ID from DynamoDB"
            print(f"✗ {error_msg}")
            sys.exit(1)

        # Step 12: Evaluate XGBoost model
        print(f"\n[12/13] Evaluating XGBoost model...")
        xgboost_rmsle = evaluate_xgboost_model(model, X, y, feature_names, lambdas['lmbda_sales'])
        if xgboost_rmsle is None:
            error_msg = "Failed to evaluate XGBoost model"
            print(f"✗ {error_msg}")
            sys.exit(1)
        
        # Step 13: Save evaluation results to DynamoDB
        print(f"\n[13/13] Saving evaluation results to DynamoDB...")
        save_evaluation_results(
            dynamodb_resource,
            job_table_name,
            model_job_id=model_job_id,
            year=year,
            biweek_num=biweek_num,
            results={
            'sarimax': evaluation_results,
            'xgbsr': {'rmsle': Decimal(str(xgboost_rmsle))}
        })

        print("\n" + "=" * 70)
        print("✓ Evaluation completed successfully!")
        print("=" * 70)
        sys.exit(0)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
