import json
import logging

import aio_pika

from core.config import settings
from core.exceptions import NonRetryableIngestionError, RetryableIngestionError
from infrastructure.redis_client import get_redis
from services.ingestion_service import run_ingestion_pipeline, send_ingestion_callback

logger = logging.getLogger(__name__)


class VideoIndexWorker:
    async def process(self, message: aio_pika.IncomingMessage) -> None:
        # requeue=False: on unhandled exception aio_pika NACKs without requeue,
        # which triggers RabbitMQ's dead-letter routing to the DLQ automatically.
        async with message.process(requeue=False):
            payload = json.loads(message.body)
            yt_video_id = payload["youtubeVideoId"]
            idempotency_key = payload["idempotencyKey"]
            attempt = payload.get("attemptNumber", 1)
            start_stage = payload.get("startFromStage", "TRANSCRIPT_EXTRACTION")

            # videoId is optional — used only for logging (the webhook is keyed by youtubeVideoId).
            video_id = payload.get("videoId")
            # chatbotId is the Qdrant tenant key. Fall back to trainingSourceId only for
            # backward compatibility with messages published before Spring Boot added it.
            chatbot_id = payload.get("chatbotId") or str(payload["trainingSourceId"])
            video_title = payload.get("videoTitle")

            # startFromStage is accepted for diagnostics but NOT acted upon: intermediate
            # results are not cached between attempts, so every attempt re-runs all stages.
            logger.info(
                "job received videoId=%s youtubeVideoId=%s chatbotId=%s attempt=%s startFromStage=%s",
                video_id, yt_video_id, chatbot_id, attempt, start_stage,
            )

            # Idempotency check — skip if this exact attempt was already indexed
            try:
                redis = await get_redis()
                redis_key = f"indexed:{idempotency_key}"
                if await redis.exists(redis_key):
                    logger.info(
                        "duplicate idempotencyKey=%s — reporting COMPLETED without reprocessing",
                        idempotency_key,
                    )
                    await send_ingestion_callback(yt_video_id, {
                        "status": "COMPLETED",
                        "attemptsMade": attempt,
                    })
                    return
            except Exception as exc:
                logger.warning(
                    "Redis idempotency check failed, proceeding without it: %s", exc
                )
                redis = None
                redis_key = None

            # Run pipeline
            try:
                await run_ingestion_pipeline(
                    chatbot_id, yt_video_id, video_title=video_title
                )

                # Mark idempotency key before sending callback so a webhook timeout
                # doesn't lead to duplicate reprocessing on the next message delivery.
                if redis and redis_key:
                    try:
                        await redis.set(redis_key, "1", ex=settings.IDEMPOTENCY_TTL_SECONDS)
                    except Exception as exc:
                        logger.warning("Redis mark failed: %s", exc)

                await send_ingestion_callback(yt_video_id, {
                    "status": "COMPLETED",
                    "attemptsMade": attempt,
                })

            except NonRetryableIngestionError as exc:
                logger.error(
                    "permanent failure videoId=%s stage=%s reason=%s",
                    video_id, exc.stage, exc.reason,
                )
                await send_ingestion_callback(yt_video_id, {
                    "status": "FAILED",
                    "failedStage": exc.stage,
                    "failureReason": exc.user_message,
                    "attemptsMade": attempt,
                    "retryable": False,
                })

            except RetryableIngestionError as exc:
                retryable = attempt < settings.MAX_RETRIES
                logger.warning(
                    "transient failure videoId=%s stage=%s attempt=%s retryable=%s reason=%s",
                    video_id, exc.stage, attempt, retryable, exc.reason,
                )
                await send_ingestion_callback(yt_video_id, {
                    "status": "FAILED",
                    "failedStage": exc.stage,
                    "failureReason": exc.user_message,
                    "attemptsMade": attempt,
                    "retryable": retryable,
                })
            # Any other exception propagates out of message.process(), triggering
            # NACK-without-requeue → automatic routing to replime.video.index.dlq.
