"""Unit tests for rag/text/text_normalizer.py — Arabic normalization."""
from rag.text.text_normalizer import normalize_arabic


def test_alef_variants_collapse_to_bare_alef():
    assert normalize_arabic("أإآٱ") == "اااا"


def test_ya_and_hamza_variants():
    assert normalize_arabic("ى") == "ي"
    assert normalize_arabic("ئ") == "ي"


def test_ta_marbuta_becomes_ha():
    assert normalize_arabic("ة") == "ه"


def test_waw_hamza_becomes_waw():
    assert normalize_arabic("ؤ") == "و"


def test_tatweel_stripped():
    assert normalize_arabic("مـــرحبا") == "مرحبا"


def test_diacritics_stripped():
    assert normalize_arabic("مُحَمَّد") == "محمد"


def test_full_word_normalization():
    assert normalize_arabic("الْأَمْثِلَة") == "الامثله"


def test_idempotent():
    text = "الأمثلة المُهِمّة"
    once = normalize_arabic(text)
    assert normalize_arabic(once) == once
