"""Worker/queue tests for the RabbitMQ consumer and DLQ handler.

aio_pika ``IncomingMessage`` is faked: ``message.process()`` returns an async
context manager and ``message.body`` carries the JSON payload.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import workers.dlq_handler as dlqh
import workers.video_index_worker as viw
from core.exceptions import NonRetryableIngestionError
from tests.factories import AsyncCM


def _message(body: dict, headers=None):
    msg = MagicMock()
    msg.body = json.dumps(body).encode()
    msg.headers = headers or {}
    msg.process = MagicMock(return_value=AsyncCM())
    return msg


def _job(**overrides):
    data = dict(youtubeVideoId="v1", idempotencyKey="k1", chatbotId="cb", attemptNumber=1)
    data.update(overrides)
    return data


async def test_duplicate_idempotency_key_reports_completed_without_processing(monkeypatch):
    redis = MagicMock()
    redis.exists = AsyncMock(return_value=True)
    monkeypatch.setattr(viw, "get_redis", AsyncMock(return_value=redis))
    pipeline = AsyncMock()
    monkeypatch.setattr(viw, "run_ingestion_pipeline", pipeline)
    callback = AsyncMock()
    monkeypatch.setattr(viw, "send_ingestion_callback", callback)

    await viw.VideoIndexWorker().process(_message(_job()))

    pipeline.assert_not_called()
    assert callback.await_args.args[1]["status"] == "COMPLETED"


async def test_new_message_runs_pipeline_and_reports_completed(monkeypatch):
    redis = MagicMock()
    redis.exists = AsyncMock(return_value=False)
    redis.set = AsyncMock()
    monkeypatch.setattr(viw, "get_redis", AsyncMock(return_value=redis))
    pipeline = AsyncMock()
    monkeypatch.setattr(viw, "run_ingestion_pipeline", pipeline)
    callback = AsyncMock()
    monkeypatch.setattr(viw, "send_ingestion_callback", callback)

    await viw.VideoIndexWorker().process(_message(_job()))

    pipeline.assert_awaited_once()
    assert callback.await_args.args[1]["status"] == "COMPLETED"


async def test_nonretryable_failure_reports_permanent_failure(monkeypatch):
    redis = MagicMock()
    redis.exists = AsyncMock(return_value=False)
    redis.set = AsyncMock()
    monkeypatch.setattr(viw, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(
        viw, "run_ingestion_pipeline",
        AsyncMock(side_effect=NonRetryableIngestionError("TRANSCRIPT_EXTRACTION", "no transcript")),
    )
    callback = AsyncMock()
    monkeypatch.setattr(viw, "send_ingestion_callback", callback)

    await viw.VideoIndexWorker().process(_message(_job()))

    payload = callback.await_args.args[1]
    assert payload["status"] == "FAILED"
    assert payload["retryable"] is False


async def test_dlq_handler_reports_dead(monkeypatch):
    callback = AsyncMock()
    monkeypatch.setattr(dlqh, "send_ingestion_callback", callback)
    msg = _message(
        {"youtubeVideoId": "v1", "videoId": 10, "attemptNumber": 3},
        headers={"x-death": [{"reason": "rejected"}]},
    )
    await dlqh.DLQHandler().handle(msg)
    payload = callback.await_args.args[1]
    assert payload["status"] == "DEAD"
    assert payload["retryable"] is False
