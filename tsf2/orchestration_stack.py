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
        task_env_vars = [
            tasks.TaskEnvironmentVariable(name="ENV", value=env_name),
            tasks.TaskEnvironmentVariable(name="DATA_BUCKET", value=data_bucket.bucket_name),
            tasks.TaskEnvironmentVariable(name="MODEL_BUCKET", value=model_bucket.bucket_name),
            tasks.TaskEnvironmentVariable(name="JOB_TABLE", value=job_table.table_name),
            tasks.TaskEnvironmentVariable(name="MODEL_TABLE", value=model_table.table_name),
            tasks.TaskEnvironmentVariable(name="DATE", value=sfn.JsonPath.string_at('$.date')),
            tasks.TaskEnvironmentVariable(name="YEAR", value=str(sfn.JsonPath.number_at('$.year'))),
            tasks.TaskEnvironmentVariable(name="BIWEEK_NUM", value=str(sfn.JsonPath.number_at('$.biweek_num'))),
            tasks.TaskEnvironmentVariable(name="BIWEEK_START", value=sfn.JsonPath.string_at('$.biweek_start')),
            tasks.TaskEnvironmentVariable(name="BIWEEK_END", value=sfn.JsonPath.string_at('$.biweek_end'))
        ]

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
                    environment=task_env_vars
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
                    environment=task_env_vars
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
                    environment=task_env_vars
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
                'date': sfn.JsonPath.string_at('$.date'),
                'year': sfn.JsonPath.number_at('$.year'),
                'biweek_num': sfn.JsonPath.number_at('$.biweek_num'),
                'biweek_start': sfn.JsonPath.string_at('$.biweek_start'),
                'biweek_end': sfn.JsonPath.string_at('$.biweek_end')
            }),
            result_path=sfn.JsonPath.DISCARD,
            integration_pattern=sfn.IntegrationPattern.REQUEST_RESPONSE,
        )

        # Execution tracking table (for automatic execution infrastructure)
        exec_table = dynamodb.Table(self, "ExecutionTable",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            removal_policy=removal,
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl"
        )

        # Manually construct state machine ARN (known before creation)
        state_machine_name = f"{env_name}-tsf2-state-machine"
        state_machine_arn = f"arn:aws:states:{self.region}:{self.account}:stateMachine:{state_machine_name}"

        # Completion Lambda
        completion_lambda_log_group = logs.LogGroup(self, "CompletionLambdaLogGroup",
            log_group_name=f"/aws/lambda/{env_name}-tsf2-completion",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal
        )

        completion_lambda = _lambda.Function(self, "CompletionLambda",
            function_name=f"{env_name}-tsf2-completion",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=_lambda.Code.from_inline("""import json
import os
import boto3
from botocore.exceptions import ClientError

sfn = boto3.client('stepfunctions')
dynamodb = boto3.resource('dynamodb')

def handler(event, context):
    table = dynamodb.Table(os.environ['EXEC_TABLE'])
    current_date = event.get('date')
    
    if not current_date:
        print("No date in event, skipping")
        return {'statusCode': 200}
    
    try:
        # Delete current job
        table.delete_item(
            Key={'pk': 'jobs', 'sk': current_date},
            ConditionExpression='attribute_exists(sk)'
        )
        print(f"Completed and removed job for {current_date}")
    except ClientError as e:
        if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
            print(f"Error deleting job: {e}")
        # Continue even if delete fails (idempotent)
    
    try:
        # Query next pending job (earliest date)
        response = table.query(
            KeyConditionExpression='pk = :pk',
            FilterExpression='#status = :status',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':pk': 'jobs', ':status': 'pending'},
            Limit=1,
            ConsistentRead=True
        )
        
        if response['Items']:
            next_job = response['Items'][0]
            job_data = next_job.get('data', {})
            next_date = next_job['sk']
            
            print(f"Starting next job for {next_date}")
            sfn.start_execution(
                stateMachineArn=os.environ['STATE_MACHINE_ARN'],
                input=json.dumps(job_data)
            )
        else:
            print("No pending jobs in queue")
    except ClientError as e:
        print(f"Error querying next job: {e}")
    
    return {'statusCode': 200}"""),
            environment={
                "STATE_MACHINE_ARN": state_machine_arn,
                "EXEC_TABLE": exec_table.table_name
            },
            log_group=completion_lambda_log_group,
        )

        # Completion task to dequeue and process next job
        completion_task = tasks.LambdaInvoke(self, "CompletionTask",
            lambda_function=completion_lambda,
            payload=sfn.TaskInput.from_object({
                'date': sfn.JsonPath.string_at('$.date')
            }),
            result_path=sfn.JsonPath.DISCARD,
            integration_pattern=sfn.IntegrationPattern.REQUEST_RESPONSE,
        )

        # Chain preprocessing to parallel tasks to completion
        preprocessing_task.next(parallel).next(completion_task)

        # State machine
        self.state_machine = sfn.StateMachine(self, "TSF2StateMachine",
            state_machine_name=state_machine_name,
            definition_body=sfn.DefinitionBody.from_chainable(preprocessing_task),
            logs=sfn.LogOptions(
                destination=sm_log_group,
                level=sfn.LogLevel.ALL
            ),
            tracing_enabled=True
        )

        # Completion Lambda IAM grants
        exec_table.grant_read_write_data(completion_lambda)
        self.state_machine.grant_start_execution(completion_lambda)

        # ── Step Functions IAM grants ──
        # evaluation_task_def.task_role.grant_pass_role(evaluation_task_def.task_role)
        data_bucket.grant_read(self.state_machine)
        model_bucket.grant_read_write(self.state_machine)
        job_table.grant_read_write_data(self.state_machine)
        model_table.grant_read_write_data(self.state_machine)

        # ── Automatic execution infrastructure ──
        # SQS queue for S3 events
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
from botocore.exceptions import ClientError

sfn = boto3.client('stepfunctions')
dynamodb = boto3.resource('dynamodb')

def is_trigger_date(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d')
    last_day = monthrange(d.year, d.month)[1]
    return d.day == 15 or d.day == last_day
                                          
def calculate_biweek(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d')
    month = d.month
    day = d.day
    biweek_in_month = 1 if day <= 15 else 2
    biweek_start = datetime(d.year, month, 1) if biweek_in_month == 1 else datetime(d.year, month, 16)
    biweek_end = datetime(d.year, month, 15) if biweek_in_month == 1 else datetime(d.year, month, monthrange(d.year, month)[1])
    return d.year, ((month - 1) * 2) + biweek_in_month, biweek_start, biweek_end

def handler(event, context):
    table = dynamodb.Table(os.environ['EXEC_TABLE'])
    
    for record in event['Records']:
        body = json.loads(record['body'])
        s3_detail = body.get('detail', {})
        s3_key = s3_detail.get('object', {}).get('key', '')
        
        try:
            key = s3_key.split('/')
            date_str = '-'.join(key[2:5])
            if not (is_trigger_date(date_str) and key[-1] == 'train.csv'):
                continue
            
            year, biweek_num, biweek_start, biweek_end = calculate_biweek(date_str)
            job_data = {
                'date': date_str,
                'year': year,
                'biweek_num': biweek_num,
                'biweek_start': biweek_start.strftime('%Y-%m-%d'),
                'biweek_end': biweek_end.strftime('%Y-%m-%d')
            }
            
            # Check if job already running for this date
            try:
                response = table.get_item(
                    Key={'pk': 'jobs', 'sk': date_str},
                    ConsistentRead=True
                )
                if 'Item' in response:
                    print(f"Job for {date_str} already exists, skipping")
                    continue
            except ClientError as e:
                print(f"Error checking job: {e}")
                continue
            
            # Atomically insert job with conditional write
            try:
                table.put_item(
                    Item={
                        'pk': 'jobs',
                        'sk': date_str,
                        'status': 'pending',
                        'data': job_data
                    },
                    ConditionExpression='attribute_not_exists(sk)'
                )
                print(f"Queued job for {date_str}")
                
                # Check if this is the only job (queue was empty)
                response = table.scan(
                    FilterExpression='#status = :status',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={':status': 'pending'},
                    Select='COUNT',
                    ConsistentRead=True
                )
                
                if response['Count'] == 1:  # Only our newly inserted job
                    print(f"Queue was empty, starting execution for {date_str}")
                    sfn.start_execution(
                        stateMachineArn=os.environ['STATE_MACHINE_ARN'],
                        input=json.dumps(job_data)
                    )
                else:
                    print(f"Queue has {response['Count']} pending jobs, queueing only")
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                    print(f"Job for {date_str} already queued/running, skipping")
                else:
                    print(f"Error inserting job: {e}")
                    
        except (IndexError, ValueError) as e:
            print(f"Could not extract date from key: {s3_key}, error: {e}")
    
    return {'statusCode': 200}"""),
            environment={
                "STATE_MACHINE_ARN": self.state_machine.state_machine_arn,
                "EXEC_TABLE": exec_table.table_name
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
        exec_table.grant_read_write_data(trigger_lambda)
        trigger_queue.grant_consume_messages(trigger_lambda)
        self.state_machine.grant_start_execution(trigger_lambda)
