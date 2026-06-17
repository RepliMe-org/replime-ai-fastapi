import logging
from datetime import datetime, timezone

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from services.http_client import get_http_client, is_retryable_http_error

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@retry(
    retry=retry_if_exception(is_retryable_http_error),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=2),
    reraise=True,
)
async def _send_event(payload: dict) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/internal/analytics/event"
    response = await get_http_client().post(
        url,
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_TOKEN},
    )
    response.raise_for_status()


async def report_content_gap(chatbot_id: str, query: str, language: str) -> None:
    """Record that a user asked an in-domain question the content does not cover."""
    try:
        await _send_event({
            "type": "CONTENT_GAP",
            "chatbotId": chatbot_id,
            "query": query,
            "language": language,
            "timestamp": _now_iso(),
        })
    except Exception as exc:
        logger.warning("report_content_gap failed chatbot_id=%s: %s", chatbot_id, exc)


async def report_video_cited(
    chatbot_id: str,
    youtube_video_id: str,
    video_title: str,
    query: str,
) -> None:
    """Record that a video was cited in an answer."""
    try:
        await _send_event({
            "type": "VIDEO_CITED",
            "chatbotId": chatbot_id,
            "youtubeVideoId": youtube_video_id,
            "videoTitle": video_title,
            "query": query,
            "timestamp": _now_iso(),
        })
    except Exception as exc:
        logger.warning("report_video_cited failed chatbot_id=%s: %s", chatbot_id, exc)
