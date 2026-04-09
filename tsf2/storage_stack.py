from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    CfnOutput,
    RemovalPolicy,
)
from constructs import Construct

class StorageStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        removal = RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY

        # Data storage bucket
        self.bucket = s3.Bucket(self, "Tsf2DataBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=removal,
            auto_delete_objects=(env_name != "prod")
        )
        CfnOutput(self, "DataBucketName", 
            value=self.bucket.bucket_name,
            export_name=f"{env_name}-DataBucketName"
        )

        # Model storage bucket
        self.model_bucket = s3.Bucket(self, "Tsf2ModelBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=removal,
            auto_delete_objects=(env_name != "prod")
        )
        CfnOutput(self, "ModelBucketName",
            value=self.model_bucket.bucket_name,
            export_name=f"{env_name}-ModelBucketName"
        )

        # DynamoDB table for current model pointer
        self.model_table = dynamodb.Table(self, "Tsf2ModelTable",
            table_name="Tsf2ModelPointer",
            partition_key=dynamodb.Attribute(name="model_id", type=dynamodb.AttributeType.STRING),
            removal_policy=removal,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )
        CfnOutput(self, "ModelTableName",
            value=self.model_table.table_name,
            export_name=f"{env_name}-ModelTableName"
        )
    
        # DynamoDB table for job metadata
        self.job_table = dynamodb.Table(self, "Tsf2JobTable",
            table_name=f"{env_name}-Tsf2JobMetadata",
            partition_key=dynamodb.Attribute(name="job_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="run_date", type=dynamodb.AttributeType.STRING),
            removal_policy=removal,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )
        CfnOutput(self, "JobTableName",
            value=self.job_table.table_name,
            export_name=f"{env_name}-JobTableName"
        )
