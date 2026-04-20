#!/usr/bin/env python3
import os

import aws_cdk as cdk

from tsf2.storage_stack import StorageStack


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

app.synth()
