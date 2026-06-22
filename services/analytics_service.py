import json
import logging
import re

from core.config import settings
from rag import prompts
from rag.llm_client import LLMClient
from schemas.analytics import (
    AnalyticsRequest,
    AnalyticsResponse,
    ContentGapCluster,
    QuestionCluster,
)

logger = logging.getLogger(__name__)

# Bound the prompt size — a representative sample is enough for theme clustering.
_MAX_QUESTIONS = 300


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
    # Nothing to cluster — return early without an LLM call.
    if not req.questions:
        return AnalyticsResponse()

    questions = req.questions[:_MAX_QUESTIONS]
    question_lines = "\n".join(
        f"- [{'answered' if q.answered_with_sources else 'unanswered'}] {q.text}"
        for q in questions
    )
    user_content = (
        f"CHANNEL DESCRIPTION:\n{req.description or '(none yet)'}\n\n"
        f"AUDIENCE QUESTIONS ({len(questions)}):\n{question_lines}"
    )
    messages = [
        {"role": "system", "content": prompts.ANALYTICS},
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
        executive_summary=data.get("executiveSummary", "") or "",
    )
