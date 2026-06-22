"""Unit tests for the enqueue-job Lambda handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_s3_event


@pytest.fixture
def enqueue_module():
    import enqueue_job.app as module

    return module


@pytest.fixture
def mocks(enqueue_module):
    mock_sqs = MagicMock()
    mock_lambda = MagicMock()
    with (
        patch.object(enqueue_module, "sqs", mock_sqs),
        patch.object(enqueue_module, "lambda_client", mock_lambda),
        patch.object(enqueue_module, "s3", MagicMock()),
    ):
        yield {
            "sqs": mock_sqs,
            "lambda": mock_lambda,
            "s3": enqueue_module.s3,
        }


def test_handler_ignores_non_daily_upload(enqueue_module, mocks):
    result = enqueue_module.handler(make_s3_event("processed/output.csv"), None)
    assert result == {"enqueued": False, "reason": "not_a_daily_upload"}
    mocks["sqs"].send_message.assert_not_called()


def test_handler_ignores_pre_cutoff_upload(enqueue_module, mocks):
    result = enqueue_module.handler(
        make_s3_event("raw/daily/2017/06/30/train.csv"),
        None,
    )
    assert result == {"enqueued": False, "reason": "before_cutoff"}
    mocks["sqs"].send_message.assert_not_called()


def test_handler_waits_for_incomplete_biweek(enqueue_module, mocks, july_2017_first_half_period):
    with patch.object(enqueue_module, "biweek_data_is_complete", return_value=False):
        result = enqueue_module.handler(
            make_s3_event("raw/daily/2017/07/10/train.csv"),
            None,
        )

    assert result == {"enqueued": False, "reason": "biweek_incomplete"}
    mocks["sqs"].send_message.assert_not_called()


def test_handler_skips_already_processed_biweek(enqueue_module, mocks, july_2017_first_half_period):
    with (
        patch.object(enqueue_module, "biweek_data_is_complete", return_value=True),
        patch.object(enqueue_module, "biweek_already_processed", return_value=True),
    ):
        result = enqueue_module.handler(
            make_s3_event("raw/daily/2017/07/10/train.csv"),
            None,
        )

    assert result == {"enqueued": False, "reason": "already_processed"}
    mocks["sqs"].send_message.assert_not_called()


def test_handler_enqueues_complete_biweek_and_invokes_starter(enqueue_module, mocks, july_2017_first_half_period):
    with (
        patch.object(enqueue_module, "biweek_data_is_complete", return_value=True),
        patch.object(enqueue_module, "biweek_already_processed", return_value=False),
    ):
        result = enqueue_module.handler(
            make_s3_event("raw/daily/2017/07/10/train.csv"),
            None,
        )

    assert result == {"enqueued": True, "biweek": "2017-BW-13"}
    mocks["sqs"].send_message.assert_called_once_with(
        QueueUrl=enqueue_module.QUEUE_URL,
        MessageBody=json.dumps(july_2017_first_half_period.to_payload()),
        MessageGroupId="tsf2-pipeline",
        MessageDeduplicationId="2017-BW-13",
    )
    mocks["lambda"].invoke.assert_called_once_with(
        FunctionName="test-starter",
        InvocationType="Event",
        Payload=json.dumps({"source": "enqueue_job"}),
    )
