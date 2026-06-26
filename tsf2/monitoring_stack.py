from aws_cdk import (
    Duration,
    Stack,
    aws_cloudwatch as cloudwatch,
)
from constructs import Construct

METRIC_NAMESPACE = "TSF2/Evaluation"
METRIC_NAME = "XGBoostRMSLE"


class MonitoringStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        rmsle_metric = cloudwatch.Metric(
            namespace=METRIC_NAMESPACE,
            metric_name=METRIC_NAME,
            dimensions_map={"Environment": env_name},
            statistic="Maximum",
            period=Duration.minutes(5),
        )

        dashboard = cloudwatch.Dashboard(
            self,
            "EvaluationDashboard",
            dashboard_name=f"{env_name}-tsf2-evaluation",
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="XGBoost RMSLE",
                width=24,
                left=[rmsle_metric],
            ),
            cloudwatch.SingleValueWidget(
                title="Latest XGBoost RMSLE",
                width=12,
                metrics=[rmsle_metric],
            ),
        )
