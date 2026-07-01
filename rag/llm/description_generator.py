import logging
from functools import lru_cache

from core.config import settings
from rag.llm import prompts
from rag.llm.llm_client import LLMClient, get_client_for

logger = logging.getLogger(__name__)

_LANGUAGE_NAME = {"ar": "Arabic", "en": "English"}


class DescriptionGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def regenerate(self, video_samples: list[dict], language: str) -> str:
        """Regenerate the channel description from a per-video sample.

        ``video_samples`` is ``[{"video_title": str, "excerpts": list[str]}, ...]``
        (see VectorStore.sample_chunks). Titles give topic breadth across the whole
        channel; excerpts add detail. Derived fresh from current Qdrant state, so
        topics from deleted/removed videos disappear.

        ``language`` is the chatbot's dominant corpus language ("ar"/"en"), used as
        the output language — authoritative (derived from the stored per-chunk
        ``content_language``), not guessed from a truncated excerpt sample.
        """
        titles = [
            v["video_title"].strip()
            for v in video_samples
            if v.get("video_title", "").strip()
        ]
        excerpts: list[str] = []
        for v in video_samples:
            excerpts.extend(v.get("excerpts", []))
        combined = "\n\n".join(excerpts)[:settings.DESCRIPTION_SAMPLE_MAX_CHARS]

        language_name = _LANGUAGE_NAME.get(language, language)
        titles_block = "\n".join(f"- {t}" for t in titles) if titles else "(none provided)"
        user_content = (
            f"VIDEO TITLES:\n{titles_block}\n\n"
            f"TRANSCRIPT EXCERPTS:\n{combined or '(none)'}"
        )
        messages = [
            {"role": "system", "content": prompts.DESCRIPTION.format(language_name=language_name)},
            {"role": "user", "content": user_content},
        ]
        result, _ = await self._llm_client.generate(messages, max_tokens=256, temperature=0.2)
        return result.strip()


@lru_cache(maxsize=1)
def get_description_generator() -> DescriptionGenerator:
    return DescriptionGenerator(get_client_for(settings.DESCRIPTION_MODEL))
