from core.chroma_client import get_vector_store
from core.exceptions import RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)


def retrieve_chunks(question: str, influencer_username: str, k: int = 4) -> list[str]:
    """
    Given a question and an influencer username,
    return the top-k most relevant text chunks from their ChromaDB collection.
    Each influencer's chunks are stored in a collection named after their username.
    """
    logger.info(f"Retrieving top-{k} chunks for '{influencer_username}' | question: {question}")

    try:
        vector_store = get_vector_store(collection_name=influencer_username)
        results = vector_store.similarity_search(question, k=k)
    except Exception as e:
        logger.error(f"Retrieval failed for '{influencer_username}': {e}")
        raise RetrievalError(f"Could not retrieve chunks for '{influencer_username}'") from e

    if not results:
        logger.warning(f"No chunks found in collection '{influencer_username}'")
        return []

    chunks = [doc.page_content for doc in results]
    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks
