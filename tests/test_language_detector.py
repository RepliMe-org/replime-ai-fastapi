"""Unit tests for rag/text/language_detector.py — Arabic vs English detection."""
from rag.text.language_detector import _arabic_letter_ratio, detect_language
from schemas.chat import ConversationMessage


def test_empty_text_returns_fallback():
    assert detect_language("") == "en"
    assert detect_language("   ", fallback="ar") == "ar"


def test_arabic_dominant_text_detected():
    assert detect_language("ما هو الذكاء الاصطناعي وكيف يعمل في الحياة") == "ar"


def test_english_text_detected():
    assert detect_language("What is artificial intelligence and how does it work") == "en"


def test_arabic_letter_ratio():
    assert _arabic_letter_ratio("hello world") == 0.0
    assert _arabic_letter_ratio("مرحبا بالعالم") == 1.0
    # Digits and punctuation are not letters, so they don't count in the ratio.
    assert _arabic_letter_ratio("123 !!!") == 0.0


def test_history_fallback_infers_arabic():
    history = [ConversationMessage(role="USER", content="ما هو الذكاء الاصطناعي")]
    # A letterless query can't be detected on its own → falls back to history.
    assert detect_language("12345", history=history) == "ar"


def test_history_fallback_stops_at_english_user_message():
    history = [ConversationMessage(role="USER", content="hello there my friend")]
    assert detect_language("12345", history=history) == "en"
