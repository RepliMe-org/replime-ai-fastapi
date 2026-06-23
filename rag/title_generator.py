import logging
from functools import lru_cache

from core.config import settings
from rag import prompts
from rag.llm_client import LLMClient, get_client_for

logger = logging.getLogger(__name__)


class TitleGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def generate(self, query: str) -> str | None:
        messages = [
            {"role": "system", "content": prompts.TITLE},
            {"role": "user", "content": query},
        ]
        try:
            title, _ = await self._llm_client.generate(messages, max_tokens=32, temperature=0.3)
            return title.strip() or None
        except Exception as exc:
            logger.warning("Title generation failed: %s", exc)
            return None


@lru_cache(maxsize=1)
def get_title_generator() -> TitleGenerator:
    return TitleGenerator(get_client_for(settings.TITLE_MODEL))
