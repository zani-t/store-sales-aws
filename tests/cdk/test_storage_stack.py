"""CDK assertion tests for the storage stack."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.assertions as assertions
import pytest

from tsf2.storage_stack import StorageStack


@pytest.fixture
def storage_template():
    app = cdk.App()
    stack = StorageStack(app, "test-storage", env_name="dev")
    return assertions.Template.from_stack(stack)


def test_storage_stack_creates_data_and_model_buckets(storage_template):
    storage_template.resource_count_is("AWS::S3::Bucket", 2)
    storage_template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [
                    {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            },
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
        },
    )


def test_storage_stack_enables_eventbridge_on_data_bucket(storage_template):
    storage_template.has_resource_properties(
        "Custom::S3BucketNotifications",
        {"NotificationConfiguration": {"EventBridgeConfiguration": {}}},
    )


def test_storage_stack_creates_model_and_job_tables(storage_template):
    storage_template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
            "KeySchema": [{"AttributeName": "model", "KeyType": "HASH"}],
        },
    )
    storage_template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
            "KeySchema": [
                {"AttributeName": "job_type", "KeyType": "HASH"},
                {"AttributeName": "complete_timestamp", "KeyType": "RANGE"},
            ],
        },
    )


def test_dev_storage_stack_uses_destroy_policies():
    app = cdk.App()
    stack = StorageStack(app, "dev-storage", env_name="dev")
    template = assertions.Template.from_stack(stack)

    template.has_resource(
        "Custom::S3AutoDeleteObjects",
        {"DeletionPolicy": "Delete", "UpdateReplacePolicy": "Delete"},
    )
    template.has_resource(
        "AWS::DynamoDB::Table",
        {"DeletionPolicy": "Delete", "UpdateReplacePolicy": "Delete"},
    )


def test_prod_storage_stack_retains_resources():
    app = cdk.App()
    stack = StorageStack(app, "prod-storage", env_name="prod")
    template = assertions.Template.from_stack(stack)

    template.has_resource(
        "AWS::S3::Bucket",
        {"DeletionPolicy": "Retain", "UpdateReplacePolicy": "Retain"},
    )
    template.has_resource(
        "AWS::DynamoDB::Table",
        {"DeletionPolicy": "Retain", "UpdateReplacePolicy": "Retain"},
    )
    template.resource_count_is("Custom::S3AutoDeleteObjects", 0)
