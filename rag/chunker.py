import bisect

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.exceptions import TranscriptError
from rag.text_normalizer import normalize_arabic

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 100
_ARABIC_CHUNK_SIZE = 900
_ARABIC_CHUNK_OVERLAP = 180
_MIN_SEGMENT_LENGTH = 25


def _merge_short_segments(segments: list[dict]) -> list[dict]:
    """Merge segments shorter than _MIN_SEGMENT_LENGTH chars into the previous one."""
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        if len(seg["text"].strip()) < _MIN_SEGMENT_LENGTH:
            merged[-1]["text"] = merged[-1]["text"] + " " + seg["text"]
        else:
            merged.append(dict(seg))
    return merged


def chunk_transcript(
    segments: list[dict],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    language: str = "en",
) -> list[dict]:
    if not segments:
        return []

    if language == "ar":
        chunk_size = _ARABIC_CHUNK_SIZE
        chunk_overlap = _ARABIC_CHUNK_OVERLAP
        segments = _merge_short_segments(segments)
        segments = [
            {**seg, "text": normalize_arabic(seg["text"])} for seg in segments
        ]

    offsets: list[int] = []
    start_times: list[float] = []
    parts: list[str] = []
    cursor = 0

    for seg in segments:
        offsets.append(cursor)
        start_times.append(seg.get("start"))
        parts.append(seg["text"])
        cursor += len(seg["text"]) + 1

    full_text = " ".join(parts)

    if not full_text.strip():
        raise TranscriptError("Transcript text is empty — cannot chunk")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(full_text)

    result: list[dict] = []
    # Known limitation: str.find() returns first occurrence, so repeated
    # phrases may get an incorrect timestamp. Acceptable for MVP.
    for chunk_text in chunks:
        pos = full_text.find(chunk_text)
        seg_index = bisect.bisect_right(offsets, pos) - 1
        seg_index = max(seg_index, 0)
        raw = start_times[seg_index]
        timestamp = int(raw) if raw is not None else None
        result.append({"text": chunk_text, "timestamp_seconds": timestamp})

    return result
