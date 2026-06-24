#!/usr/bin/env python3
import os

import aws_cdk as cdk

from tsf2.storage_stack import StorageStack
from tsf2.compute_stack import ComputeStack
from tsf2.orchestration_stack import OrchestrationStack
from tsf2.monitoring_stack import MonitoringStack


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

orchestration_stack = OrchestrationStack(
    app, f"{env_name}-OrchestrationStack",
    env_name=env_name,
    data_bucket=storage_stack.data_bucket,
    model_bucket=storage_stack.model_bucket,
    job_table=storage_stack.job_table,
    model_table=storage_stack.model_table,
    preprocessing_lambda=compute_stack.preprocessing_lambda,
    cluster=compute_stack.cluster,
    evaluation_task_def=compute_stack.evaluation_task_def,
    evaluation_container=compute_stack.evaluation_container,
    smx_retraining_task_def=compute_stack.smx_retraining_task_def,
    smx_container=compute_stack.smx_container,
    xgbsr_retraining_task_def=compute_stack.xgbsr_retraining_task_def,
    xgbsr_container=compute_stack.xgbsr_container,
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
        )
    )

MonitoringStack(
    app, f"{env_name}-MonitoringStack",
    env_name=env_name,
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
        )
    )

app.synth()
