"""Enqueue completed biweeks onto the FIFO job queue."""

from __future__ import annotations

import json
import os

import boto3

from shared.biweek import (
    biweek_already_processed,
    biweek_data_is_complete,
    get_biweek_for_date,
    parse_event_date,
)

sqs = boto3.client("sqs")
s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

QUEUE_URL = os.environ["QUEUE_URL"]
DATA_BUCKET = os.environ["DATA_BUCKET"]
STARTER_FUNCTION_NAME = os.environ["STARTER_FUNCTION_NAME"]
MESSAGE_GROUP_ID = os.environ.get("MESSAGE_GROUP_ID", "tsf2-pipeline")


def _invoke_starter() -> None:
    lambda_client.invoke(
        FunctionName=STARTER_FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps({"source": "enqueue_job"}),
    )


def handler(event, _context):
    upload_date = parse_event_date(event)
    if upload_date is None:
        print("Ignoring event without a raw/daily object key")
        return {"enqueued": False, "reason": "not_a_daily_upload"}

    period = get_biweek_for_date(upload_date)
    if period is None:
        print(f"Ignoring pre-cutoff upload date {upload_date}")
        return {"enqueued": False, "reason": "before_cutoff"}

    if not biweek_data_is_complete(s3, DATA_BUCKET, period):
        print(
            f"Biweek {period.year}/BW-{period.biweek_num} is not complete yet "
            f"(triggered by {upload_date})"
        )
        return {"enqueued": False, "reason": "biweek_incomplete"}

    if biweek_already_processed(s3, DATA_BUCKET, period):
        print(f"Biweek {period.year}/BW-{period.biweek_num} already processed")
        return {"enqueued": False, "reason": "already_processed"}

    payload = period.to_payload()
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(payload),
        MessageGroupId=MESSAGE_GROUP_ID,
        MessageDeduplicationId=period.deduplication_id,
    )
    print(f"Enqueued biweek {period.deduplication_id}: {payload}")
    _invoke_starter()
    return {"enqueued": True, "biweek": period.deduplication_id}
