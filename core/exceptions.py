class TranscriptNotFoundError(Exception):
    """YouTube video has no captions/transcript."""
    pass


class EmptyTranscriptError(Exception):
    """Transcript was loaded but came back empty."""
    pass


class ChunkingError(Exception):
    """Failed to split transcript into chunks."""
    pass


