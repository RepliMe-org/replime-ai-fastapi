from schemas.chat import ChatbotConfig


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


def build_system_prompt(config: ChatbotConfig, language: str) -> str:
    if config.talk_like_me:
        persona = (
            f"You are {config.chatbot_name}.\n"
            f"{config.persona_description}\n"
            "Study the provided context excerpts and mirror the creator's exact voice, vocabulary, and phrasing."
        )
    else:
        tone_instruction = TONE_MAP.get(config.tone.upper() if config.tone else "", "adapt to the context")
        formality_instruction = FORMALITY_MAP.get(config.formality.upper() if config.formality else "", "adapt to the context")
        persona = (
            f"You are {config.chatbot_name}.\n"
            f"{config.persona_description}\n"
            f"Tone: {tone_instruction}\n"
            f"{formality_instruction}"
        )

    verbosity_instruction = VERBOSITY_MAP.get(config.verbosity.upper(), "")

    rules = (
        "\n\n---\n"
        "Only answer based on the provided context from the creator's videos.\n"
        "If the context does not contain enough information to answer, say so honestly.\n"
        "Never make up information not present in the context.\n"
        f"Always reply in the language with code: {language}"
    )

    return f"{persona}\n{verbosity_instruction}{rules}"

