from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.exceptions import ChunkingError
from core.logger import get_logger

logger = get_logger(__name__)

def split_transcript(transcript: str) -> list[str]:
    logger.info(f"Splitting transcript of {len(transcript)} characters")

    if not transcript or not transcript.strip():
        raise ChunkingError("Transcript is empty, nothing to split")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_text(transcript)
    except Exception as e:
        raise ChunkingError(str(e)) from e

    if not chunks:
        raise ChunkingError("Splitter returned no chunks")

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks