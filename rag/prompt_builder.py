from core.exceptions import InvalidRequestError
from schemas.chat import ChatbotConfig, ConversationMessage

_ROLE_MAP = {"USER": "user", "BOT": "assistant"}


def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        timestamp = chunk.get("timestamp_seconds")
        mmss = f"{timestamp // 60:02d}:{timestamp % 60:02d}" if timestamp is not None else "00:00"
        t_param = f"&t={timestamp}s" if timestamp is not None else ""
        parts.append(
            f"[Source {i}] {chunk['video_title']} @ {mmss}\n"
            f"{chunk['chunk_text']}\n"
            f"Link: https://youtube.com/watch?v={chunk['youtube_video_id']}{t_param}"
        )
    return "\n---\n".join(parts)


def build_prompt(
    query: str,
    chunks: list[dict],
    history: list[ConversationMessage],
    config: ChatbotConfig,
) -> list[dict]:
    keywords = ", ".join(config.persona_keywords)
    formatted_chunks = _format_chunks(chunks)

    system_content = (
        f"You are {config.chatbot_name}.\n"
        f"{config.persona_description}\n"
        f"Tone: {config.tone}. Style: {config.response_length}.\n"
        f"Keywords: {keywords}\n"
        f"\n--- RELEVANT VIDEO CONTENT ---\n"
        f"{formatted_chunks}\n"
        f"\n--- INSTRUCTIONS ---\n"
        f"1. Answer ONLY from the video content above.\n"
        f"2. Keep your {config.tone} tone throughout.\n"
        f"3. Include YouTube links with timestamps when referencing content.\n"
        f"4. Never invent information not in the excerpts."
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    recent_history = history[-config.max_context_turns:]
    for msg in recent_history:
        role = _ROLE_MAP.get(msg.role)
        if role is None:
            raise InvalidRequestError(f"Unknown role: {msg.role!r}")
        messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": query})

    return messages
