from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ec2 as ec2,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_dynamodb as dynamodb,
    RemovalPolicy,
    Duration,
)
from constructs import Construct

class PreprocessingStack(Stack):

    def __init__(self, scope: Construct, construct_id: str,
                 env_name: str,
                 data_bucket: s3.IBucket,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        removal = RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY

        # Lambda function log group
        preprocessing_log_group = logs.LogGroup(self, "PreprocessingLogGroup",
            log_group_name=f"/aws/lambda/{env_name}-tsf2-preprocessing",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal
        )

        # Lambda function
        self.preprocessing_lambda = _lambda.DockerImageFunction(self, "PreprocessingLambda",
            function_name=f"{env_name}-tsf2-preprocessing",
            code=_lambda.DockerImageCode.from_image_asset("containers/preprocessing"),
            memory_size=3008,
            timeout=Duration.minutes(15),
            environment={
                "ENV": env_name,
                "DATA_BUCKET": data_bucket.bucket_name,
            },
            log_group=preprocessing_log_group,
        )

        # ── Preproessing IAM grants ──
        # List bucket files on raw prefix
        self.preprocessing_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[data_bucket.bucket_arn],
            conditions={
                "StringLike": {
                    "s3:prefix": [
                        "raw/*",
                        "processed/*",
                    ]
                }
            }
        ))
        # Read raw data and marker files (HeadObject) on both prefixes
        self.preprocessing_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[
                f"{data_bucket.bucket_arn}/raw/*",
                f"{data_bucket.bucket_arn}/processed/*"
            ]
        ))
        # Write to processed prefix only
        self.preprocessing_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:PutObject"],
            resources=[f"{data_bucket.bucket_arn}/processed/*"]
        ))
        