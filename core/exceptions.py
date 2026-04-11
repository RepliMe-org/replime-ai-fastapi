class TranscriptNotFoundError(Exception):
    """YouTube video has no captions/transcript."""
    pass


class EmptyTranscriptError(Exception):
    """Transcript was loaded but came back empty."""
    pass


class ChunkingError(Exception):
    """Failed to split transcript into chunks."""
    pass


class EmbeddingError(Exception):
    """Failed to load or run the embedding model."""
    pass


class VectorStoreConnectionError(Exception):
    """Failed to connect to the ChromaDB vector store."""
    pass


class IngestionError(Exception):
    """Failed to store chunks into the vector store."""
    pass


