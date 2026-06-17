import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from rag.llm_client import LLMClient
from rag.profile_store import get_profile_store
from schemas.analytics import (
    AnalyticsRequest,
    AnalyticsResponse,
    CitedVideoStat,
    ContentGapCluster,
    QuestionCluster,
)
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


# ── Batch analytics (POST /ai/analytics/process) ──────────────────────────────

# Bound the prompt size — a representative sample is enough for theme clustering.
_MAX_QUESTIONS = 300
_MAX_GAPS = 200
_TOP_VIDEOS = 10

_ANALYTICS_SYSTEM_PROMPT = (
    "You are a content analytics assistant for a creator's Q&A chatbot. "
    "You receive the audience's questions, the questions the content could NOT answer "
    "(content gaps), and a profile of what the channel covers. Analyze them and return "
    "insights as STRICT JSON only — no prose, no code fences.\n\n"
    "JSON schema:\n"
    "{\n"
    '  "mostAskedClusters": [{"theme": str, "count": int, "exampleQuestions": [str]}],\n'
    '  "contentGaps": [{"topic": str, "frequency": int, "sampleQuestions": [str]}],\n'
    '  "executiveSummary": str,\n'
    '  "contentOpportunities": [str]\n'
    "}\n\n"
    "Rules:\n"
    "- Group similar questions into themes; count = how many provided questions fall in each.\n"
    "- exampleQuestions/sampleQuestions: up to 3 verbatim from the input.\n"
    "- contentOpportunities: concrete video/topic ideas the creator should make next, "
    "derived from the gaps and under-covered themes.\n"
    "- Write themes, summary, and opportunities in the dominant language of the input.\n"
    "- Return at most 8 clusters and 8 gap topics, ordered by count/frequency descending."
)


def _aggregate_cited_videos(req: AnalyticsRequest) -> list[CitedVideoStat]:
    counts: dict[str, int] = defaultdict(int)
    titles: dict[str, str] = {}
    for cv in req.cited_videos:
        counts[cv.youtube_video_id] += cv.citation_count
        titles[cv.youtube_video_id] = cv.video_title
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:_TOP_VIDEOS]
    return [
        CitedVideoStat(
            youtube_video_id=vid,
            video_title=titles.get(vid, vid),
            citation_count=count,
        )
        for vid, count in ranked
    ]


def _parse_llm_json(raw: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = raw.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the substring between the first { and last }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse analytics LLM output as JSON")
    return {}


_analytics_client: LLMClient | None = None


def _get_analytics_client() -> LLMClient:
    global _analytics_client
    if _analytics_client is None:
        _analytics_client = LLMClient(settings.ANALYTICS_MODEL)
    return _analytics_client


async def compute_analytics(req: AnalyticsRequest) -> AnalyticsResponse:
    # Video citation ranking is deterministic — computed without the LLM.
    most_cited = _aggregate_cited_videos(req)

    # Nothing to cluster — return early without an LLM call.
    if not req.questions and not req.content_gaps:
        return AnalyticsResponse(most_cited_videos=most_cited)

    profile = await asyncio.to_thread(get_profile_store().get_profile, req.chatbot_id)

    questions = req.questions[:_MAX_QUESTIONS]
    gaps = [g.query for g in req.content_gaps[:_MAX_GAPS]]
    user_content = (
        f"CHANNEL PROFILE:\n{profile or '(none yet)'}\n\n"
        f"AUDIENCE QUESTIONS ({len(questions)}):\n"
        + "\n".join(f"- {q}" for q in questions)
        + f"\n\nUNANSWERED (CONTENT GAP) QUESTIONS ({len(gaps)}):\n"
        + "\n".join(f"- {q}" for q in gaps)
    )
    messages = [
        {"role": "system", "content": _ANALYTICS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw, _ = await _get_analytics_client().generate(
            messages, max_tokens=1500, temperature=0.3
        )
        data = _parse_llm_json(raw)
    except Exception as exc:
        logger.exception("compute_analytics LLM call failed: %s", exc)
        data = {}

    clusters = [
        QuestionCluster(
            theme=c.get("theme", ""),
            count=int(c.get("count", 0) or 0),
            example_questions=c.get("exampleQuestions", []) or [],
        )
        for c in data.get("mostAskedClusters", [])
        if isinstance(c, dict)
    ]
    gap_clusters = [
        ContentGapCluster(
            topic=g.get("topic", ""),
            frequency=int(g.get("frequency", 0) or 0),
            sample_questions=g.get("sampleQuestions", []) or [],
        )
        for g in data.get("contentGaps", [])
        if isinstance(g, dict)
    ]

    return AnalyticsResponse(
        most_asked_clusters=clusters,
        content_gaps=gap_clusters,
        most_cited_videos=most_cited,
        executive_summary=data.get("executiveSummary", "") or "",
        content_opportunities=data.get("contentOpportunities", []) or [],
    )
