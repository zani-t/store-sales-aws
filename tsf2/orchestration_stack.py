from aws_cdk import (
    Stack,
    CfnOutput,
    aws_iam as iam,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda as _lambda,
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

from tsf2.lambda_bundle import lambda_asset_code

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
            tasks.TaskEnvironmentVariable(name="YEAR", value=f"{sfn.JsonPath.number_at('$.year')}"),
            tasks.TaskEnvironmentVariable(name="BIWEEK_NUM", value=f"{sfn.JsonPath.number_at('$.biweek_num')}"),
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

        # Chain preprocessing to parallel tasks
        preprocessing_task.next(parallel)

        state_machine_name = f"{env_name}-tsf2-state-machine"

        # State machine
        self.state_machine = sfn.StateMachine(self, "TSF2StateMachine",
            state_machine_name=state_machine_name,
            definition_body=sfn.DefinitionBody.from_chainable(preprocessing_task),
            logs=sfn.LogOptions(
                destination=sm_log_group,
                level=sfn.LogLevel.ALL
            ),
            tracing_enabled=True,
        )

        # ── Step Functions IAM grants ──
        # evaluation_task_def.task_role.grant_pass_role(evaluation_task_def.task_role)
        data_bucket.grant_read(self.state_machine)
        model_bucket.grant_read_write(self.state_machine)
        job_table.grant_read_write_data(self.state_machine)
        model_table.grant_read_write_data(self.state_machine)

        # ── Serialized job queue (FIFO) ──
        # Messages stay invisible for the full pipeline duration; the starter deletes
        # them only after Step Functions reaches a terminal status.
        self.job_queue = sqs.Queue(self, "JobQueue",
            queue_name=f"{env_name}-tsf2-jobs.fifo",
            fifo=True,
            content_based_deduplication=False,
            visibility_timeout=Duration.hours(2),
            retention_period=Duration.days(14),
            removal_policy=removal,
        )

        self.orchestration_lock_table = dynamodb.Table(self, "OrchestrationLockTable",
            table_name=f"{env_name}-tsf2-orchestration-lock",
            partition_key=dynamodb.Attribute(name="id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=removal,
        )

        lambda_code = lambda_asset_code()
        message_group_id = "tsf2-pipeline"

        starter_log_group = logs.LogGroup(self, "StarterLogGroup",
            log_group_name=f"/aws/lambda/{env_name}-tsf2-starter",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal,
        )

        self.starter_lambda = _lambda.Function(self, "StarterLambda",
            function_name=f"{env_name}-tsf2-starter",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="starter.app.handler",
            code=lambda_code,
            timeout=Duration.minutes(1),
            reserved_concurrent_executions=1,
            environment={
                "STATE_MACHINE_ARN": self.state_machine.state_machine_arn,
                "QUEUE_URL": self.job_queue.queue_url,
                "LOCK_TABLE_NAME": self.orchestration_lock_table.table_name,
            },
            log_group=starter_log_group,
        )

        enqueue_log_group = logs.LogGroup(self, "EnqueueJobLogGroup",
            log_group_name=f"/aws/lambda/{env_name}-tsf2-enqueue-job",
            retention=logs.RetentionDays.ONE_MONTH if env_name == "prod" else logs.RetentionDays.ONE_WEEK,
            removal_policy=removal,
        )

        self.enqueue_lambda = _lambda.Function(self, "EnqueueJobLambda",
            function_name=f"{env_name}-tsf2-enqueue-job",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="enqueue_job.app.handler",
            code=lambda_code,
            timeout=Duration.minutes(1),
            environment={
                "QUEUE_URL": self.job_queue.queue_url,
                "DATA_BUCKET": data_bucket.bucket_name,
                "STARTER_FUNCTION_NAME": self.starter_lambda.function_name,
                "MESSAGE_GROUP_ID": message_group_id,
            },
            log_group=enqueue_log_group,
        )

        # ── IAM grants for queue orchestration ──
        self.job_queue.grant_send_messages(self.enqueue_lambda)
        self.job_queue.grant_consume_messages(self.starter_lambda)
        self.orchestration_lock_table.grant_read_write_data(self.starter_lambda)
        self.state_machine.grant_start_execution(self.starter_lambda)
        self.state_machine.grant_read(self.starter_lambda)
        data_bucket.grant_read(self.enqueue_lambda)
        self.starter_lambda.grant_invoke(self.enqueue_lambda)

        # ── Automatic execution infrastructure ──
        # S3 Object Created → enqueue when a biweek is complete
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
        s3_event_rule.add_target(targets.LambdaFunction(self.enqueue_lambda))

        # Step Functions terminal status → delete queue message and drain next job
        completion_rule = events.Rule(self, "StateMachineCompletionRule",
            event_pattern=events.EventPattern(
                source=["aws.states"],
                detail_type=["Step Functions Execution Status Change"],
                detail={
                    "status": ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"],
                    "stateMachineArn": [self.state_machine.state_machine_arn],
                }
            )
        )
        completion_rule.add_target(targets.LambdaFunction(self.starter_lambda))

        CfnOutput(self, "JobQueueUrl",
            value=self.job_queue.queue_url,
            export_name=f"{env_name}-JobQueueUrl",
        )
        CfnOutput(self, "StateMachineArn",
            value=self.state_machine.state_machine_arn,
            export_name=f"{env_name}-StateMachineArn",
        )
