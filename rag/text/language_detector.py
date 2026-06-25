import logging
import re

from langdetect import LangDetectException, detect_langs
from langdetect.detector_factory import DetectorFactory

DetectorFactory.seed = 0  # deterministic results across calls

logger = logging.getLogger(__name__)

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_SUPPORTED = frozenset({"en", "ar"})
_CONFIDENCE_THRESHOLD = 0.85


def detect_language(
    text: str,
    history: list | None = None,
    fallback: str = "en",
) -> str:
    text = text.strip()
    if not text:
        return fallback

    # Fast path: presence of Arabic Unicode block characters
    if _ARABIC_RE.search(text):
        return "ar"

    # langdetect with confidence threshold
    try:
        results = detect_langs(text)
        if results:
            top = results[0]
            if top.prob >= _CONFIDENCE_THRESHOLD and top.lang in _SUPPORTED:
                return top.lang
    except LangDetectException as exc:
        logger.warning("Language detection failed: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected error during language detection: %s", exc)

    # Fallback: infer from most recent user message in conversation history
    if history:
        for msg in reversed(history):
            if getattr(msg, "role", None) == "USER" and msg.content.strip():
                if _ARABIC_RE.search(msg.content):
                    return "ar"
                break

    return fallback
