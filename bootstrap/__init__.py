import boto3
from botocore.exceptions import ClientError
import sys

MARKER = '_COMPLETE'

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
