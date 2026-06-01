#!/usr/bin/env python3
import os

import aws_cdk as cdk

from tsf2.storage_stack import StorageStack
from tsf2.compute_stack import ComputeStack
from tsf2.preprocessing_stack import PreprocessingStack


app = cdk.App()
env_name = app.node.try_get_context("env") or "dev"

storage_stack = StorageStack(
    app, f"{env_name}-StorageStack",
    env_name=env_name,
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
        )
    )

compute_stack = ComputeStack(
    app, f"{env_name}-ComputeStack",
    env_name=env_name,
    data_bucket=storage_stack.data_bucket,
    model_bucket=storage_stack.model_bucket,
    job_table=storage_stack.job_table,
    model_table=storage_stack.model_table,
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
        )
    )
compute_stack.add_dependency(storage_stack)

preprocessing_stack = PreprocessingStack(
    app, f"{env_name}-PreprocessingStack",
    env_name=env_name,
    data_bucket=storage_stack.data_bucket,
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
        )
    )
preprocessing_stack.add_dependency(storage_stack)

app.synth()
