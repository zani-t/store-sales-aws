from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda as _lambda,
    aws_lambda_event_sources as lambda_event_sources,
    aws_ecs as ecs,
    aws_logs as logs,
    aws_s3 as s3,
    aws_sqs as sqs,
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as targets,
    Duration,
    RemovalPolicy,
)
from constructs import Construct

class OrchestrationStack(Stack):

    def __init__(self, scope: Construct, construct_id: str,
                 env_name: str,
                 data_bucket: s3.IBucket,
                 model_bucket: s3.IBucket,
                 job_table: dynamodb.ITable,
                 model_table: dynamodb.ITable,
                 preprocessing_lambda: _lambda.DockerImageFunction,
                 cluster: ecs.ICluster,
                 evaluation_task_def: ecs.FargateTaskDefinition,
                 evaluation_container: ecs.ContainerDefinition,
                 smx_retraining_task_def: ecs.FargateTaskDefinition,
                 smx_container: ecs.ContainerDefinition,
                 xgbsr_retraining_task_def: ecs.FargateTaskDefinition,
                 xgbsr_container: ecs.ContainerDefinition,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        removal = RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY

        # State Machine Log Group
        sm_log_group = logs.LogGroup(self, "StateMachineLogGroup",
            log_group_name=f"/aws/vendedlogs/states/{env_name}-tsf2-state-machine",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal
        )

        # ── Task definitions ──
        # Evaluation Task
        evaluation_task = tasks.EcsRunTask(self, "EvaluationTask",
            cluster=cluster,
            task_definition=evaluation_task_def,
            launch_target=tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            container_overrides=[
                tasks.ContainerOverride(
                    container_definition=evaluation_container,
                    environment=[
                        tasks.TaskEnvironmentVariable(name="ENV", value=env_name),
                        tasks.TaskEnvironmentVariable(name="DATA_BUCKET", value=data_bucket.bucket_name),
                        tasks.TaskEnvironmentVariable(name="MODEL_BUCKET", value=model_bucket.bucket_name),
                        tasks.TaskEnvironmentVariable(name="JOB_TABLE", value=job_table.table_name),
                        tasks.TaskEnvironmentVariable(name="MODEL_TABLE", value=model_table.table_name),
                    ]
                )
            ],
            assign_public_ip=True,
            result_path=sfn.JsonPath.DISCARD,
        )

        # SMX Retraining Task
        smx_retraining_task = tasks.EcsRunTask(self, "SMXRetrainingTask",
            cluster=cluster,
            task_definition=smx_retraining_task_def,
            launch_target=tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            container_overrides=[
                tasks.ContainerOverride(
                    container_definition=smx_container,
                    environment=[
                        tasks.TaskEnvironmentVariable(name="ENV", value=env_name),
                        tasks.TaskEnvironmentVariable(name="DATA_BUCKET", value=data_bucket.bucket_name),
                        tasks.TaskEnvironmentVariable(name="MODEL_BUCKET", value=model_bucket.bucket_name),
                        tasks.TaskEnvironmentVariable(name="JOB_TABLE", value=job_table.table_name),
                        tasks.TaskEnvironmentVariable(name="MODEL_TABLE", value=model_table.table_name),
                    ]
                )
            ],
            assign_public_ip=True,
            result_path=sfn.JsonPath.DISCARD,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
        )

        # XGBSR Retraining Task
        xgbsr_retraining_task = tasks.EcsRunTask(self, "XGBRetrainingTask",
            cluster=cluster,
            task_definition=xgbsr_retraining_task_def,
            launch_target=tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST
            ),
            container_overrides=[
                tasks.ContainerOverride(
                    container_definition=xgbsr_container,
                    environment=[
                        tasks.TaskEnvironmentVariable(name="ENV", value=env_name),
                        tasks.TaskEnvironmentVariable(name="DATA_BUCKET", value=data_bucket.bucket_name),
                        tasks.TaskEnvironmentVariable(name="MODEL_BUCKET", value=model_bucket.bucket_name),
                        tasks.TaskEnvironmentVariable(name="JOB_TABLE", value=job_table.table_name),
                        tasks.TaskEnvironmentVariable(name="MODEL_TABLE", value=model_table.table_name),
                    ]
                )
            ],
            assign_public_ip=True,
            result_path=sfn.JsonPath.DISCARD,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
        )

        # ── State machine provision ──
        # Failure handling
        sarimax_failed = sfn.Pass(self, "SarimaxFailed",
            result=sfn.Result.from_object({"sarimax_status": "failed", "xgboost_status": "skipped"}),
            result_path="$.retraining_result"
        )

        # Retraining branch chain
        retraining_branch = smx_retraining_task.add_catch(
            sarimax_failed,
            errors=["States.ALL"],
            result_path="$.sarimax_error"
        ).next(xgbsr_retraining_task)

        # Parallel state
        parallel = sfn.Parallel(self, "EvaluationAndRetraining")
        parallel.branch(evaluation_task)
        parallel.branch(retraining_branch)

        # Preprocessing task
        preprocessing_task = tasks.LambdaInvoke(self, "PreprocessingTask",
            lambda_function=preprocessing_lambda,
            payload=sfn.TaskInput.from_object({
                'date': sfn.JsonPath.string_at('$.date')
            }),
            result_path=sfn.JsonPath.DISCARD,
            integration_pattern=sfn.IntegrationPattern.REQUEST_RESPONSE,
        )

        # Chain preprocessing to parallel tasks
        preprocessing_task.next(parallel)

        # State machine
        self.state_machine = sfn.StateMachine(self, "TSF2StateMachine",
            state_machine_name=f"{env_name}-tsf2-state-machine",
            definition_body=sfn.DefinitionBody.from_chainable(preprocessing_task),
            logs=sfn.LogOptions(
                destination=sm_log_group,
                level=sfn.LogLevel.ALL
            ),
            tracing_enabled=True
        )

        # ── Step Functions IAM grants ──
        # evaluation_task_def.task_role.grant_pass_role(evaluation_task_def.task_role)
        data_bucket.grant_read(self.state_machine)
        model_bucket.grant_read_write(self.state_machine)
        job_table.grant_read_write_data(self.state_machine)
        model_table.grant_read_write_data(self.state_machine)

        # ── Automatic execution infrastructure ──
        # SQS queue
        trigger_queue = sqs.Queue(self, "TriggerQueue",
            queue_name=f"{env_name}-tsf2-trigger",
            visibility_timeout=Duration.minutes(15),
            retention_period=Duration.days(7),
            removal_policy=removal
        )

        # S3 event notification
        s3_event_rule = events.Rule(self, "S3RawDailyEventRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [data_bucket.bucket_name]},
                    "object": {"key": [{"prefix": "raw/daily/"}]}
                }
            )
        )
        s3_event_rule.add_target(targets.SqsQueue(trigger_queue))

        # Lambda trigger for SQS messages
        trigger_lambda_log_group = logs.LogGroup(self, "TriggerLambdaLogGroup",
            log_group_name=f"/aws/lambda/{env_name}-tsf2-trigger",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal
        )

        trigger_lambda = _lambda.Function(self, "TriggerLambda",
            function_name=f"{env_name}-tsf2-trigger",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="index.handler",
            reserved_concurrent_executions=1,
            code=_lambda.Code.from_inline("""import json
import os
import boto3
from datetime import datetime
from calendar import monthrange

sfn = boto3.client('stepfunctions')

def is_trigger_date(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d')
    last_day = monthrange(d.year, d.month)[1]
    return d.day == 15 or d.day == last_day

def handler(event, context):
    for record in event['Records']:
        body = json.loads(record['body'])
        # EventBridge events come through SQS as detail field
        s3_detail = body.get('detail', {})
        s3_key = s3_detail.get('object', {}).get('key', '')
        # Extract date from raw/daily/YYYY/MM/DD/...
        try:
            key = s3_key.split('/')
            date_str = '-'.join(key[2:5])
            if is_trigger_date(date_str) and key[-1] == 'train.csv':
                try:
                    sfn.describe_execution(
                        executionArn=f"{os.environ['STATE_MACHINE_ARN'].replace(':stateMachine:', ':execution:')}:{date_str}"
                    )
                    print(f"Execution for {date_str} already exists, skipping")
                    continue
                except sfn.exceptions.ExecutionDoesNotExist:
                    print(f"Starting execution for date {date_str}")
                    pass
                sfn.start_execution(
                    stateMachineArn=os.environ['STATE_MACHINE_ARN'],
                    name=date_str,  # Use date as execution name
                    input=json.dumps({'date': date_str})
                )
        except (IndexError, ValueError):
            print(f"Could not extract date from key: {s3_key}")
    return {'statusCode': 200}"""),
            environment={
                "STATE_MACHINE_ARN": self.state_machine.state_machine_arn
            },
            log_group=trigger_lambda_log_group,
        )

        # Trigger Lambda event source
        trigger_lambda.add_event_source(
            lambda_event_sources.SqsEventSource(trigger_queue,
                batch_size=1
            )
        )

        # IAM grants
        trigger_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["states:DescribeExecution"],
            resources=[f"arn:aws:states:{self.region}:{self.account}:execution:{self.state_machine.state_machine_name}:*"]
        ))
        trigger_queue.grant_consume_messages(trigger_lambda)
        self.state_machine.grant_start_execution(trigger_lambda)
