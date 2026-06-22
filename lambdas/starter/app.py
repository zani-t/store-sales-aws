"""Drain the FIFO queue one Step Functions execution at a time."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

sfn = boto3.client("stepfunctions")
sqs = boto3.client("sqs")
dynamodb = boto3.resource("dynamodb")

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]
QUEUE_URL = os.environ["QUEUE_URL"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

LOCK_ID = "job-runner"
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}


def _lock_table():
    return dynamodb.Table(LOCK_TABLE_NAME)


def _get_lock() -> dict:
    response = _lock_table().get_item(Key={"id": LOCK_ID})
    return response.get("Item") or {}


def _execution_is_terminal(execution_arn: str) -> bool:
    status = sfn.describe_execution(executionArn=execution_arn)["status"]
    return status in TERMINAL_STATUSES


def _recover_stale_lock(lock: dict) -> bool:
    """Clear orphaned locks left behind by partial starter runs."""
    execution_arn = lock.get("execution_arn")
    if not execution_arn:
        receipt_handle = lock.get("receipt_handle")
        if receipt_handle:
            _release_message({"ReceiptHandle": receipt_handle})
        _clear_lock()
        print("Cleared lock that never recorded a Step Functions execution")
        return True

    if not _execution_is_terminal(execution_arn):
        return False

    receipt_handle = lock.get("receipt_handle")
    if receipt_handle:
        try:
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
        except ClientError as error:
            print(f"Could not delete queue message during stale-lock recovery: {error}")

    _clear_lock()
    print(f"Cleared stale lock for finished execution {execution_arn}")
    return True


def _clear_lock() -> None:
    _lock_table().delete_item(Key={"id": LOCK_ID})


def _try_acquire_lock(message: dict[str, Any], payload: dict[str, Any]) -> bool:
    table = _lock_table()
    try:
        table.put_item(
            Item={
                "id": LOCK_ID,
                "is_running": True,
                "message_id": message["MessageId"],
                "receipt_handle": message["ReceiptHandle"],
                "payload": json.dumps(payload),
            },
            ConditionExpression="attribute_not_exists(id)",
        )
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def _release_message(message: dict[str, Any]) -> None:
    sqs.change_message_visibility(
        QueueUrl=QUEUE_URL,
        ReceiptHandle=message["ReceiptHandle"],
        VisibilityTimeout=0,
    )


def _start_next_execution() -> dict[str, Any]:
    lock = _get_lock()
    if lock.get("is_running"):
        if _recover_stale_lock(lock):
            lock = {}
        else:
            print("An execution is already running; skipping drain")
            return {"started": False, "reason": "already_running"}

    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=0,
    )
    messages = response.get("Messages") or []
    if not messages:
        print("Queue is empty")
        return {"started": False, "reason": "queue_empty"}

    message = messages[0]
    payload = json.loads(message["Body"])

    if not _try_acquire_lock(message, payload):
        print("Lost race to acquire lock; releasing message visibility")
        _release_message(message)
        return {"started": False, "reason": "lock_race"}

    biweek_label = f"{payload['year']}-BW-{payload['biweek_num']}"
    execution_name = f"{biweek_label}-{int(time.time())}"[:80]
    try:
        execution = sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps(payload),
        )
    except ClientError as error:
        _clear_lock()
        _release_message(message)
        raise error

    _lock_table().update_item(
        Key={"id": LOCK_ID},
        UpdateExpression="SET execution_arn = :execution_arn",
        ExpressionAttributeValues={":execution_arn": execution["executionArn"]},
    )
    print(f"Started execution {execution['executionArn']} for {biweek_label}")
    return {"started": True, "executionArn": execution["executionArn"]}


def _handle_completion(event: dict[str, Any]) -> dict[str, Any]:
    detail = event.get("detail") or {}
    status = detail.get("status")
    execution_arn = detail.get("executionArn")

    if status not in TERMINAL_STATUSES:
        print(f"Ignoring non-terminal status {status}")
        return {"handled": False, "reason": "non_terminal_status"}

    lock = _get_lock()
    if not lock:
        print("No lock record present on completion event")
        return {"handled": False, "reason": "no_lock"}

    locked_arn = lock.get("execution_arn")
    if locked_arn and execution_arn and locked_arn != execution_arn:
        print(f"Ignoring completion for unrelated execution {execution_arn}")
        return {"handled": False, "reason": "unrelated_execution"}

    receipt_handle = lock.get("receipt_handle")
    if receipt_handle:
        sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
        print("Deleted queue message for completed execution")

    _clear_lock()
    return _start_next_execution()


def handler(event, _context):
    if event.get("source") == "aws.states":
        return _handle_completion(event)
    return _start_next_execution()
