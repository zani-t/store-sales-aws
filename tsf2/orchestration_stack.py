from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_events as events,
    aws_events_targets as targets,
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

        # State machine
        self.state_machine = sfn.StateMachine(self, "TSF2StateMachine",
            state_machine_name=f"{env_name}-tsf2-state-machine",
            definition_body=sfn.DefinitionBody.from_chainable(parallel),
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

        # ── Preprocessing Lambda configuration ──
        preprocessing_lambda.add_environment(
            "STATE_MACHINE_ARN",
            self.state_machine.state_machine_arn
        )
        self.state_machine.grant_start_execution(preprocessing_lambda)
