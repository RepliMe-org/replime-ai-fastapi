import logging

from schemas.chat import ConversationMessage
from rag.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a query rewriting assistant. "
    "Rewrite the user's latest message so it is fully self-contained — resolve any pronouns, "
    "references to previous messages, or implied context so the rewritten query makes sense "
    "without the conversation history. If the query is already self-contained, return it unchanged. "
    "Return only the rewritten query. No explanation, no added commentary."
)


def _format_history(history: list[ConversationMessage]) -> str:
    lines = []
    for msg in history:
        label = "User" if msg.role == "USER" else "Bot"
        lines.append(f"{label}: {msg.content}")
    return "\n".join(lines)


class QueryRewriter:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def rewrite(self, query: str, history: list[ConversationMessage]) -> str:
        if not history:
            return query

        formatted = _format_history(history)
        user_content = (
            f"Conversation history:\n{formatted}\n\nLatest message: {query}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        rewritten, _ = await self._llm_client.generate(
            messages, max_tokens=128, temperature=0.1
        )
        rewritten = rewritten.strip()
        logger.info(
            "Query rewritten: original=%r rewritten=%r",
            query[:80],
            rewritten[:80],
        )
        return rewritten


_query_rewriter: QueryRewriter | None = None


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter(get_llm_client())
    return _query_rewriter
