import logging

from core.config import settings
from rag import prompts
from rag.llm_client import LLMClient
from schemas.chat import ConversationMessage

logger = logging.getLogger(__name__)


def _format_history(history: list[ConversationMessage]) -> str:
    lines = []
    for msg in history:
        label = "User" if msg.role == "USER" else "Bot"
        lines.append(f"{label}: {msg.content}")
    return "\n".join(lines)


class QueryRewriter:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def rewrite(
        self,
        query: str,
        history: list[ConversationMessage],
        language: str = "en",
    ) -> str:
        if not history:
            return query

        formatted = _format_history(history)
        user_content = f"Conversation history:\n{formatted}\n\nLatest message: {query}"
        messages = [
            {"role": "system", "content": prompts.QUERY_REWRITE},
            {"role": "user", "content": user_content},
        ]
        rewritten, _ = await self._llm_client.generate(messages, max_tokens=128, temperature=0.1)
        rewritten = rewritten.strip()
        logger.info(
            "Query rewritten: original=%r rewritten=%r language=%s",
            query[:80],
            rewritten[:80],
            language,
        )
        return rewritten


_query_rewriter: QueryRewriter | None = None


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter(llm_client=LLMClient(settings.REWRITE_MODEL))
    return _query_rewriter
