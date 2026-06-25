from rag.llm import prompts
from schemas.chat import ChatbotConfig, ConversationMessage


TONE_MAP = {
    "FRIENDLY": "warm and approachable",
    "ENCOURAGING": "motivating and supportive",
    "NEUTRAL": "objective and informative",
    "HUMOROUS": "light-hearted with occasional humor",
}
VERBOSITY_MAP = {
    "CONCISE": "Keep answers brief and to the point.",
    "BALANCED": "Provide a moderate level of detail.",
    "DETAILED": "Give thorough, in-depth answers.",
}
FORMALITY_MAP = {
    "CASUAL": "Use casual, conversational language.",
    "NEUTRAL": "Use neutral language, neither too formal nor too casual.",
    "FORMAL": "Use formal, professional language.",
}

_ROLE_MAP = {"USER": "user", "BOT": "assistant"}

_LANGUAGE_NAME = {"ar": "Arabic", "en": "English"}


def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        timestamp = chunk.get("timestamp_seconds")
        mmss = f"{timestamp // 60:02d}:{timestamp % 60:02d}" if timestamp is not None else "00:00"
        t_param = f"&t={timestamp}s" if timestamp is not None else ""
        parts.append(
            f"[{i}] {chunk['video_title']} @ {mmss}\n"
            f"{chunk['chunk_text']}\n"
            f"Link: https://youtube.com/watch?v={chunk['youtube_video_id']}{t_param}"
        )
    return "\n---\n".join(parts)


def build_system_prompt(config: ChatbotConfig, language: str) -> str:
    if config.talk_like_me:
        persona_lines = [
            f"You are {config.chatbot_name}.",
            "Study the context excerpts carefully and mirror the creator's exact voice, vocabulary, and phrasing.",
        ]
    else:
        tone_instruction = TONE_MAP.get(config.tone.upper() if config.tone else "", "adapt to the context")
        formality_instruction = FORMALITY_MAP.get(config.formality.upper() if config.formality else "", "adapt to the context")
        persona_lines = [
            f"You are {config.chatbot_name}.",
            f"Tone: {tone_instruction}",
            formality_instruction,
        ]

    persona = "\n".join(line for line in persona_lines if line)

    verbosity_instruction = VERBOSITY_MAP.get(config.verbosity.upper(), "")

    rules = prompts.ANSWER_RULES.format(language_name=_LANGUAGE_NAME.get(language, language))

    parts = [persona, verbosity_instruction, rules]
    return "\n\n".join(part for part in parts if part)


def build_messages(
    query: str,
    chunks: list[dict],
    history: list[ConversationMessage],
    config: ChatbotConfig,
    language: str,
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(config, language)}
    ]

    for msg in history:
        role = _ROLE_MAP.get(msg.role)
        if role is None:
            continue
        messages.append({"role": role, "content": msg.content})

    if chunks:
        user_content = (
            f"<context>\n{_format_chunks(chunks)}\n</context>\n\n"
            f"<question>\n{query}\n</question>"
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})
    return messages