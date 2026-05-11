import re

_TATWEEL = re.compile(r"ـ")
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")
_YA_VARIANTS = re.compile(r"ى")
_TA_MARBUTA = re.compile(r"ة")
_WAW_HAMZA = re.compile(r"ؤ")
_YA_HAMZA = re.compile(r"ئ")
# Diacritics (tashkeel) + superscript alef
_DIACRITICS = re.compile(r"[ً-ٰٟ]")


def normalize_arabic(text: str) -> str:
    text = _TATWEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _YA_VARIANTS.sub("ي", text)
    text = _TA_MARBUTA.sub("ه", text)
    text = _WAW_HAMZA.sub("و", text)
    text = _YA_HAMZA.sub("ي", text)
    text = _DIACRITICS.sub("", text)
    return text
