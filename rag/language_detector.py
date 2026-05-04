import logging
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


def detect_language(text: str, fallback: str = "en") -> str:
    try:
        if not text.strip():
            logger.warning("Empty text provided, returning fallback language")
            return fallback

        lang = detect(text)

        if not lang:
            logger.warning("Detection returned empty result, returning fallback language")
            return fallback

        return lang

    except LangDetectException as e:
        logger.warning("Language detection failed: %s, returning fallback language", e)
        return fallback
    except Exception as e:
        logger.warning("Unexpected error during language detection: %s, returning fallback language", e)
        return fallback