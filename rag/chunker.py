import bisect

from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1600 chars ≈ 400 tokens at the rough 4 chars/token rule
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 100


def chunk_transcript(   
    segments: list[dict],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    if not segments:
        return []

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(full_text)

    result: list[dict] = []
    # TODO: Fix this issue
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
