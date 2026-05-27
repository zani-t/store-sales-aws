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

class ComputeStack(Stack):

    def __init__(self, scope: Construct, construct_id: str,
                 env_name: str,
                 data_bucket: s3.IBucket,
                 model_bucket: s3.IBucket,
                 job_table: dynamodb.ITable,
                 model_table: dynamodb.ITable,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        removal = RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY

        # ── Preproessing infrastructure ──
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


        # ── General compute infrastructure ──
        # VPC
        vpc = ec2.Vpc(self, "VPC", max_azs=2, cidr="10.0.0.0/16",
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    cidr_mask=24,
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC
                )
            ]
        )

        # ECS Cluster
        cluster = ecs.Cluster(self, "Cluster",
            vpc=vpc,
            enable_fargate_capacity_providers=True
        )

        # ── Evaluation infrastructure ──
        # Container log group
        evaluation_log_group = logs.LogGroup(self, "EvaluationLogGroup",
            log_group_name=f"/aws/ecs/{env_name}-tsf2-evaluation",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal
        )

        # Fargate task definition
        self.evaluation_task_def = ecs.FargateTaskDefinition(self, "EvaluationTaskDef",
            memory_limit_mib=4096,
            cpu=512,
        )

        # Container
        self.evaluation_task_def.add_container("EvaluationContainer",
            image=ecs.ContainerImage.from_asset("containers/evaluation"),
            logging=ecs.LogDriver.aws_logs(
                log_group=evaluation_log_group,
                stream_prefix="evaluation"
            ),
            environment={
                "ENV": env_name,
                "DATA_BUCKET": data_bucket.bucket_name,
                "MODEL_BUCKET": model_bucket.bucket_name,
                "JOB_TABLE": job_table.table_name,
                "MODEL_TABLE": model_table.table_name,
            }
        )

        # ── Evaluation IAM grants ──
        # Grant S3 permissions
        data_bucket.grant_read(self.evaluation_task_def.task_role)
        model_bucket.grant_read(self.evaluation_task_def.task_role)

        # Grant DynamoDB permissions
        job_table.grant_read_write_data(self.evaluation_task_def.task_role)
        model_table.grant_read_write_data(self.evaluation_task_def.task_role)

        # ── SARIMAX retraining infrastructure ──
        # Container log group
        smx_retraining_log_group = logs.LogGroup(self, "SmxRetrainingLogGroup",
            log_group_name=f"/aws/ecs/{env_name}-tsf2-smx-retraining",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal
        )

        # Fargate task definition
        self.smx_retraining_task_def = ecs.FargateTaskDefinition(self, "SmxRetrainingTaskDef",
            memory_limit_mib=4096,
            cpu=512,
        )

        # Container
        self.smx_retraining_task_def.add_container("SmxRetrainingContainer",
            image=ecs.ContainerImage.from_asset("containers/smx-training"),
            logging=ecs.LogDriver.aws_logs(
                log_group=smx_retraining_log_group,
                stream_prefix="retraining"
            ),
            environment={
                "ENV": env_name,
                "DATA_BUCKET": data_bucket.bucket_name,
                "MODEL_BUCKET": model_bucket.bucket_name,
                "JOB_TABLE": job_table.table_name,
                "MODEL_TABLE": model_table.table_name,
            }
        )

        # ── Retraining IAM grants ──
        # Grant S3 permissions
        data_bucket.grant_read_write(self.smx_retraining_task_def.task_role)
        model_bucket.grant_read_write(self.smx_retraining_task_def.task_role)

        # Grant DynamoDB permissions
        job_table.grant_read_write_data(self.smx_retraining_task_def.task_role)
        model_table.grant_read_write_data(self.smx_retraining_task_def.task_role)
        