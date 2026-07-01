import logging
import re

from langdetect import LangDetectException, detect_langs
from langdetect.detector_factory import DetectorFactory

DetectorFactory.seed = 0  # deterministic results across calls

logger = logging.getLogger(__name__)

_ARABIC_CHAR_RE = re.compile(r"[؀-ۿ]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_SUPPORTED = frozenset({"en", "ar"})
_CONFIDENCE_THRESHOLD = 0.85
_ARABIC_RATIO_THRESHOLD = 0.3  # share of Arabic letters among all letters to call text Arabic-dominant


def _arabic_letter_ratio(text: str) -> float:
    """Share of Arabic-script letters among all alphabetic characters.

    Ratio-based rather than presence-based, so a code-switched query or a
    transcript with a few Arabic names/loanwords doesn't flip the whole text
    to Arabic — only text where Arabic is the dominant script does.
    """
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    arabic_letters = _ARABIC_CHAR_RE.findall(text)
    return len(arabic_letters) / len(letters)


def detect_language(
    text: str,
    history: list | None = None,
    fallback: str = "en",
) -> str:
    text = text.strip()
    if not text:
        return fallback

    # Fast path: Arabic is the dominant script
    if _arabic_letter_ratio(text) >= _ARABIC_RATIO_THRESHOLD:
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
                if _arabic_letter_ratio(msg.content) >= _ARABIC_RATIO_THRESHOLD:
                    return "ar"
                break

    return fallback
