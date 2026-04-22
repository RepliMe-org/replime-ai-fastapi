from schemas.chat import ChatbotConfig, ConversationMessage


def _seconds_to_mmss(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        timestamp = chunk.get("timestamp_seconds")
        mmss = _seconds_to_mmss(timestamp) if timestamp is not None else "00:00"
        t_param = f"&t={timestamp}s" if timestamp is not None else ""
        parts.append(
            f"[Source {i}] {chunk['video_title']} @ {mmss}\n"
            f"{chunk['chunk_text']}\n"
            f"Link: https://youtube.com/watch?v={chunk['youtube_video_id']}{t_param}"
        )
    return "\n---\n".join(parts)


def _format_history(history: list[ConversationMessage], max_turns: int) -> str:
    recent = history[-max_turns:]
    lines = []
    for msg in recent:
        label = "USER" if msg.role == "USER" else "BOT"
        lines.append(f"{label}: {msg.content}")
    return "\n".join(lines)


def build_prompt(
    query: str,
    chunks: list[dict],
    history: list[ConversationMessage],
    config: ChatbotConfig,
) -> str:
    formatted_history = _format_history(history, config.max_context_turns)
    formatted_chunks = _format_chunks(chunks)
    keywords = ", ".join(config.persona_keywords)

    return (
        f"You are {config.chatbot_name}.\n"
        f"{config.persona_description}\n"
        f"Tone: {config.tone}. Style: {config.response_length}.\n"
        f"Keywords: {keywords}\n"
        f"\n--- CONVERSATION HISTORY ---\n"
        f"{formatted_history}\n"
        f"\n--- RELEVANT VIDEO CONTENT ---\n"
        f"{formatted_chunks}\n"
        f"\n--- USER QUESTION ---\n"
        f"{query}\n"
        f"\n--- INSTRUCTIONS ---\n"
        f"1. Answer ONLY from the video content above.\n"
        f"2. If no relevant content, say: \"I don't have information about that in {config.chatbot_name}'s content.\"\n"
        f"3. Keep your {config.tone} tone throughout.\n"
        f"4. Include YouTube links with timestamps when referencing content.\n"
        f"5. Never invent information not in the excerpts."
    )
