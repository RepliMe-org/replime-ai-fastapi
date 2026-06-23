import logging
from difflib import get_close_matches

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import settings
from rag import prompts
from rag.llm_client import get_client_for
from schemas.chat import MessageClass
from services.http_client import get_http_client, is_retryable_http_error

logger = logging.getLogger(__name__)


async def _call_classification_llm(query: str, class_names: list[str]) -> str:
    categories = ", ".join(class_names)
    messages = [
        {"role": "system", "content": prompts.CLASSIFICATION},
        {"role": "user", "content": f"Categories: {categories}\n\nMessage: {query}"},
    ]
    result, _ = await get_client_for(settings.CLASSIFICATION_MODEL).generate(messages, max_tokens=32, temperature=0.0)
    return result.strip()


def _match_class(llm_output: str, message_classes: list[MessageClass]) -> MessageClass | None:
    # Exact match (case-insensitive)
    for mc in message_classes:
        if mc.name.lower() == llm_output.lower():
            return mc
    # Fuzzy fallback
    names = [mc.name for mc in message_classes]
    matches = get_close_matches(llm_output, names, n=1, cutoff=0.6)
    if matches:
        for mc in message_classes:
            if mc.name == matches[0]:
                return mc
    return None


@retry(
    retry=retry_if_exception(is_retryable_http_error),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=2),
    reraise=True,
)
async def _send_classification(message_id: int, class_id: int) -> None:
    url = f"{settings.SPRING_BOOT_BASE_URL}/internal/messages/{message_id}"
    response = await get_http_client().put(
        url,
        json=class_id,
        headers={"X-INTERNAL-TOKEN": settings.X_INTERNAL_TOKEN},
    )
    response.raise_for_status()


async def classify_and_report(
    message_id: int,
    query: str,
    message_classes: list[MessageClass],
) -> None:
    try:
        class_names = [mc.name for mc in message_classes]
        llm_output = await _call_classification_llm(query, class_names)
        matched = _match_class(llm_output, message_classes)
        if matched is None:
            logger.warning(
                "Classification returned unmatched label=%r for message_id=%d",
                llm_output,
                message_id,
            )
            return
        logger.info(
            "Classified message_id=%d as class_id=%d class_name=%r",
            message_id,
            matched.id,
            matched.name,
        )
        await _send_classification(message_id, matched.id)
    except Exception as exc:
        logger.exception("classify_and_report failed for message_id=%d: %s", message_id, exc)
