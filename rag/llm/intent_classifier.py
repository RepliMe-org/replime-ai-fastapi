import logging
import re
from functools import lru_cache

from core.config import settings
from rag.llm import prompts
from rag.llm.llm_client import LLMClient, get_client_for

logger = logging.getLogger(__name__)

# Common injection/jailbreak patterns — caught before LLM to guarantee blocking
_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(your\s+)?(previous|prior|all|above|instructions?|rules?|prompt)"
    r"|forget\s+(your\s+)?(instructions?|rules?|prompt|everything)"
    r"|reveal\s+(your\s+)?(system\s+)?prompt"
    r"|show\s+(me\s+)?(your\s+)?(system\s+)?prompt"
    r"|what\s+(are\s+)?(your\s+)?(instructions?|rules?|prompt|system\s+prompt)"
    r"|you\s+are\s+now\s+a"
    r"|pretend\s+(you\s+are|to\s+be)"
    r"|act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(?!.*assistant)",
    re.IGNORECASE,
)

_HARDCODED_RESPONSES: dict[str, dict[str, str]] = {
    "GREETING": {
        "en": "Hello! I'm {chatbot_name}. Feel free to ask me anything about my content.",
        "ar": "مرحباً! أنا {chatbot_name}. لا تتردد في سؤالي عن أي شيء يتعلق بمحتواي.",
    },
    "SMALL_TALK": {
        "en": "I'm here to help you explore {chatbot_name}'s content. What would you like to know?",
        "ar": "أنا هنا لمساعدتك في استكشاف محتوى {chatbot_name}. ماذا تريد أن تعرف؟",
    },
    "OUT_OF_SCOPE": {
        "en": "I can only answer questions about {chatbot_name}'s content.",
        "ar": "يمكنني فقط الإجابة على الأسئلة المتعلقة بمحتوى {chatbot_name}.",
    },
    "HARMFUL": {
        "en": "I can't help with that.",
        "ar": "لا أستطيع المساعدة في ذلك.",
    },
}

_VALID_INTENTS = frozenset(_HARDCODED_RESPONSES.keys()) | {"CONTENT_QUESTION"}


class IntentClassifier:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def classify(self, query: str, description: str | None = None) -> str:
        # Rule-based pre-filter — catches obvious injections before hitting the LLM
        if _INJECTION_PATTERNS.search(query):
            logger.warning("Injection pattern detected in query=%r", query[:80])
            return "HARMFUL"

        # With a channel description we use the domain-aware prompt; without one we
        # fall back to the original behavior.
        if description:
            system_prompt = prompts.INTENT_WITH_DESCRIPTION.format(description=description)
        else:
            system_prompt = prompts.INTENT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            result, _ = await self._llm_client.generate(messages, max_tokens=16, temperature=0.0)
            intent = result.strip().upper()
            if intent not in _VALID_INTENTS:
                logger.warning("Unknown intent label=%r, defaulting to CONTENT_QUESTION", intent)
                return "CONTENT_QUESTION"
            return intent
        except Exception as exc:
            logger.warning("Intent classification failed: %s, defaulting to CONTENT_QUESTION", exc)
            return "CONTENT_QUESTION"


def get_hardcoded_response(intent: str, language: str, chatbot_name: str) -> str:
    templates = _HARDCODED_RESPONSES.get(intent, {})
    template = templates.get(language) or templates.get("en", "")
    return template.format(chatbot_name=chatbot_name)


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    return IntentClassifier(get_client_for(settings.INTENT_MODEL))
