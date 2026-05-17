import logging

from core.config import settings
from rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a session title generator for a content Q&A chatbot. "
    "Given a user's first question, write a short title of 4–7 words that captures the main topic. "
    "Focus on the subject being asked about, not the question format. "
    "Preserve the original language of the question. "
    "Return only the title. No punctuation at the end, no quotes, no explanation."
)


class TitleGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def generate(self, query: str) -> str | None:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            title, _ = await self._llm_client.generate(messages, max_tokens=32, temperature=0.3)
            return title.strip() or None
        except Exception as exc:
            logger.warning("Title generation failed: %s", exc)
            return None


_title_generator: TitleGenerator | None = None


def get_title_generator() -> TitleGenerator:
    global _title_generator
    if _title_generator is None:
        _title_generator = TitleGenerator(
            LLMClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_FAST_MODEL)
        )
    return _title_generator
