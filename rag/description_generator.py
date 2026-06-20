import logging

from core.config import settings
from rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You maintain a short description of what topics a content creator's channel covers. "
    "You are given the CURRENT description (may be empty) and a SAMPLE of text from a newly added "
    "video. Produce an UPDATED description that merges the new video's topics into the existing one.\n"
    "Rules:\n"
    "- Output ONE concise paragraph (max 120 words) listing the themes/subjects the channel covers.\n"
    "- Write in the same language as the content (Arabic content → Arabic description).\n"
    "- Do not invent topics not present in the text. Do not list video titles.\n"
    "- Keep prior topics; only add genuinely new ones. Return only the paragraph, nothing else."
)

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
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        result, _ = await self._llm_client.generate(messages, max_tokens=256, temperature=0.3)
        return result.strip()


_description_generator: DescriptionGenerator | None = None


def get_description_generator() -> DescriptionGenerator:
    global _description_generator
    if _description_generator is None:
        _description_generator = DescriptionGenerator(LLMClient(settings.DESCRIPTION_MODEL))
    return _description_generator
