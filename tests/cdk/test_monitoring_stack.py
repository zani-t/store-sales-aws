"""CDK assertion tests for the monitoring stack."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.assertions as assertions
import pytest

from tsf2.monitoring_stack import METRIC_NAME, METRIC_NAMESPACE, MonitoringStack


@pytest.fixture
def monitoring_template():
    app = cdk.App()
    stack = MonitoringStack(app, "test-monitoring", env_name="dev")
    return assertions.Template.from_stack(stack)


def test_monitoring_stack_creates_dashboard(monitoring_template):
    monitoring_template.resource_count_is("AWS::CloudWatch::Dashboard", 1)
    monitoring_template.has_resource_properties(
        "AWS::CloudWatch::Dashboard",
        {"DashboardName": "dev-tsf2-evaluation"},
    )


def test_monitoring_stack_dashboard_references_xgboost_rmsle(monitoring_template):
    monitoring_template.has_resource_properties(
        "AWS::CloudWatch::Dashboard",
        {
            "DashboardBody": assertions.Match.object_like(
                {
                    "Fn::Join": assertions.Match.array_with(
                        [
                            "",
                            assertions.Match.array_with(
                                [
                                    assertions.Match.string_like_regexp(
                                        f".*{METRIC_NAMESPACE}.*{METRIC_NAME}.*"
                                    )
                                ]
                            ),
                        ]
                    )
                }
            )
        },
    )
