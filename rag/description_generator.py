import logging
from functools import lru_cache

from core.config import settings
from rag import prompts
from rag.llm_client import LLMClient, get_client_for

logger = logging.getLogger(__name__)

# Cap how much new-video text we feed the LLM — a representative sample is enough.
_SAMPLE_CHAR_LIMIT = 4000


class DescriptionGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def update_description(self, current_description: str | None, new_chunks: list[str]) -> str:
        sample = " ".join(new_chunks)[:_SAMPLE_CHAR_LIMIT]
        user_content = (
            f"CURRENT description:\n{current_description or '(empty)'}\n\n"
            f"SAMPLE from new video:\n{sample}"
        )
        messages = [
            {"role": "system", "content": prompts.DESCRIPTION},
            {"role": "user", "content": user_content},
        ]
        result, _ = await self._llm_client.generate(messages, max_tokens=256, temperature=0.3)
        return result.strip()


@lru_cache(maxsize=1)
def get_description_generator() -> DescriptionGenerator:
    return DescriptionGenerator(get_client_for(settings.DESCRIPTION_MODEL))
