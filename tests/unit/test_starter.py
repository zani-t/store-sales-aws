"""Unit tests for the starter Lambda queue drain and lock behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError


@pytest.fixture
def starter_module():
    import starter.app as module

    return module


@pytest.fixture
def lock_table(starter_module):
    table = MagicMock()
    with patch.object(starter_module, "dynamodb") as mock_dynamodb:
        mock_dynamodb.Table.return_value = table
        yield table


def _queue_message(payload: dict) -> dict:
    return {
        "MessageId": "msg-1",
        "ReceiptHandle": "receipt-1",
        "Body": json.dumps(payload),
    }


def test_start_next_execution_returns_queue_empty(starter_module, lock_table):
    lock_table.get_item.return_value = {}

    with patch.object(starter_module, "sqs") as mock_sqs:
        mock_sqs.receive_message.return_value = {"Messages": []}
        result = starter_module._start_next_execution()

    assert result == {"started": False, "reason": "queue_empty"}


def test_start_next_execution_skips_when_lock_is_active(starter_module, lock_table):
    lock_table.get_item.return_value = {
        "Item": {
            "id": "job-runner",
            "is_running": True,
            "execution_arn": "arn:aws:states:us-east-1:123:execution:running",
        }
    }

    with (
        patch.object(starter_module, "sfn") as mock_sfn,
        patch.object(starter_module, "sqs") as mock_sqs,
        patch.object(starter_module, "_recover_stale_lock", return_value=False),
    ):
        mock_sfn.describe_execution.return_value = {"status": "RUNNING"}
        result = starter_module._start_next_execution()

    assert result == {"started": False, "reason": "already_running"}
    mock_sqs.receive_message.assert_not_called()


def test_start_next_execution_starts_execution_for_queued_message(starter_module, lock_table):
    payload = {
        "date": "2017-07-10",
        "year": "2017",
        "biweek_num": "13",
        "biweek_start": "2017-07-01",
        "biweek_end": "2017-07-15",
    }

    lock_table.get_item.return_value = {}
    lock_table.put_item.return_value = None
    lock_table.update_item.return_value = None

    with (
        patch.object(starter_module, "sqs") as mock_sqs,
        patch.object(starter_module, "sfn") as mock_sfn,
        patch.object(starter_module, "time") as mock_time,
    ):
        mock_sqs.receive_message.return_value = {"Messages": [_queue_message(payload)]}
        mock_sfn.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:new"
        }
        mock_time.time.return_value = 1_700_000_000
        result = starter_module._start_next_execution()

    assert result == {
        "started": True,
        "executionArn": "arn:aws:states:us-east-1:123:execution:new",
    }
    mock_sfn.start_execution.assert_called_once()
    lock_table.update_item.assert_called_once()


def test_start_next_execution_releases_message_on_lock_race(starter_module, lock_table):
    payload = {
        "date": "2017-07-10",
        "year": "2017",
        "biweek_num": "13",
        "biweek_start": "2017-07-01",
        "biweek_end": "2017-07-15",
    }

    lock_table.get_item.return_value = {}
    lock_table.put_item.side_effect = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException"}},
        "PutItem",
    )

    with patch.object(starter_module, "sqs") as mock_sqs:
        mock_sqs.receive_message.return_value = {"Messages": [_queue_message(payload)]}
        result = starter_module._start_next_execution()

    assert result == {"started": False, "reason": "lock_race"}
    mock_sqs.change_message_visibility.assert_called_once_with(
        QueueUrl=starter_module.QUEUE_URL,
        ReceiptHandle="receipt-1",
        VisibilityTimeout=0,
    )


def test_handle_completion_drains_next_job_after_terminal_status(starter_module, lock_table):
    lock_table.get_item.return_value = {
        "Item": {
            "id": "job-runner",
            "is_running": True,
            "execution_arn": "arn:aws:states:us-east-1:123:execution:done",
            "receipt_handle": "receipt-1",
        }
    }

    with (
        patch.object(starter_module, "sqs") as mock_sqs,
        patch.object(starter_module, "_start_next_execution", return_value={"started": False, "reason": "queue_empty"}) as mock_drain,
    ):
        result = starter_module._handle_completion(
            {
                "detail": {
                    "status": "SUCCEEDED",
                    "executionArn": "arn:aws:states:us-east-1:123:execution:done",
                }
            }
        )

    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl=starter_module.QUEUE_URL,
        ReceiptHandle="receipt-1",
    )
    lock_table.delete_item.assert_called_once_with(Key={"id": "job-runner"})
    mock_drain.assert_called_once()
    assert result == {"started": False, "reason": "queue_empty"}


def test_handle_completion_ignores_unrelated_execution(starter_module, lock_table):
    lock_table.get_item.return_value = {
        "Item": {
            "id": "job-runner",
            "is_running": True,
            "execution_arn": "arn:aws:states:us-east-1:123:execution:current",
            "receipt_handle": "receipt-1",
        }
    }

    result = starter_module._handle_completion(
        {
            "detail": {
                "status": "SUCCEEDED",
                "executionArn": "arn:aws:states:us-east-1:123:execution:other",
            }
        }
    )

    assert result == {"handled": False, "reason": "unrelated_execution"}


def test_handler_routes_step_functions_completion_events(starter_module):
    with patch.object(starter_module, "_handle_completion", return_value={"handled": True}) as mock_completion:
        result = starter_module.handler({"source": "aws.states"}, None)

    mock_completion.assert_called_once()
    assert result == {"handled": True}


def test_handler_drains_queue_for_non_completion_events(starter_module):
    with patch.object(starter_module, "_start_next_execution", return_value={"started": False, "reason": "queue_empty"}) as mock_drain:
        result = starter_module.handler({"source": "enqueue_job"}, None)

    mock_drain.assert_called_once()
    assert result == {"started": False, "reason": "queue_empty"}
