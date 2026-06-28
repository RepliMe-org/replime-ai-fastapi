import logging
from functools import lru_cache

from core.config import settings
from rag.llm import prompts
from rag.llm.llm_client import LLMClient, get_client_for

logger = logging.getLogger(__name__)


class DescriptionGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def regenerate(self, sample_chunks: list[str]) -> str:
        """Regenerate the channel description from a representative cross-video sample.

        Derives the description fresh from the sample (Qdrant is the source of
        truth), so topics from deleted or removed videos disappear instead of
        lingering in an incrementally-merged blob.
        """
        sample = "\n\n".join(sample_chunks)[:settings.DESCRIPTION_SAMPLE_MAX_CHARS]
        messages = [
            {"role": "system", "content": prompts.DESCRIPTION},
            {"role": "user", "content": f"CHANNEL EXCERPTS:\n{sample}"},
        ]
        result, _ = await self._llm_client.generate(messages, max_tokens=256, temperature=0.2)
        return result.strip()


@lru_cache(maxsize=1)
def get_description_generator() -> DescriptionGenerator:
    return DescriptionGenerator(get_client_for(settings.DESCRIPTION_MODEL))
