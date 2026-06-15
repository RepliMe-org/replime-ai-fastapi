import json
import logging

import aio_pika

from services.ingestion_service import send_ingestion_callback

logger = logging.getLogger(__name__)


class DLQHandler:
    """Handles messages routed to replime.video.index.dlq.

    Messages arrive here when the main worker raises an unhandled exception
    (i.e. something outside NonRetryableIngestionError / RetryableIngestionError,
    such as a JSON parse error or a Redis connectivity failure). The job here is
    to tell Spring Boot the video is permanently dead so it can update the DB and
    show a manual retry button in the dashboard.
    """

    async def handle(self, message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                payload = json.loads(message.body)
            except Exception:
                logger.error("DLQ message body is not valid JSON — dropping")
                return

            video_id = payload.get("videoId")
            yt_video_id = payload.get("youtubeVideoId", "")
            attempt = payload.get("attemptNumber", 0)

            # x-death header is attached by RabbitMQ when routing to DLQ
            x_death = message.headers.get("x-death", [{}])
            death_reason = x_death[0].get("reason", "unknown") if x_death else "unknown"

            logger.error(
                "DLQ message videoId=%s youtubeVideoId=%s attempts=%s reason=%s",
                video_id, yt_video_id, attempt, death_reason,
            )

            if yt_video_id:
                await send_ingestion_callback(yt_video_id, {
                    "status": "DEAD",
                    "failureReason": (
                        f"Processing permanently failed ({death_reason}) "
                        f"after {attempt} attempt(s)"
                    ),
                    "attemptsMade": attempt,
                    "retryable": False,
                })
