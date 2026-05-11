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


def build_system_prompt(config: ChatbotConfig, language: str) -> str:
    persona_desc = (config.persona_description or "").strip()

    if config.talk_like_me:
        persona_lines = [
            f"You are {config.chatbot_name}.",
            persona_desc,
            "Study the context excerpts carefully and mirror the creator's exact voice, vocabulary, and phrasing.",
        ]
    else:
        tone_instruction = TONE_MAP.get(config.tone.upper() if config.tone else "", "adapt to the context")
        formality_instruction = FORMALITY_MAP.get(config.formality.upper() if config.formality else "", "adapt to the context")
        persona_lines = [
            f"You are {config.chatbot_name}.",
            persona_desc,
            f"Tone: {tone_instruction}",
            formality_instruction,
        ]

    persona = "\n".join(line for line in persona_lines if line)

    verbosity_instruction = VERBOSITY_MAP.get(config.verbosity.upper(), "")

    rules = (
        "---\n"
        "Rules:\n"
        "- Answer only from the provided context; do not use outside knowledge.\n"
        "- Read ALL provided sources before answering; if multiple sources cover the question, synthesize them into one cohesive answer rather than stopping at the first relevant one.\n"
        "- Synthesize and explain the information in your own words; never copy or paraphrase sentences directly from the context.\n"
        "- Present ideas clearly and naturally as if explaining to someone — not quoting a transcript.\n"
        "- When citing information, reference the source number (e.g. [Source 1]).\n"
        "- If the query is too vague or short to determine what the user is asking, ask one focused clarifying question instead of answering — even if context is available.\n"
        "- If the context lacks enough information, say so honestly instead of guessing.\n"
        f"- Always reply in the language with code: {language}"
    )

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