import pytest
import pytest_asyncio

from rag.embedder import Embedder

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIM = 384


@pytest_asyncio.fixture(scope="module")
async def embedder() -> Embedder:
    return Embedder(MODEL)


@pytest.mark.asyncio
async def test_embedding_dimension(embedder: Embedder) -> None:
    vector = await embedder.embed_one("Hello, world!")
    assert len(vector) == EXPECTED_DIM


@pytest.mark.asyncio
async def test_same_input_same_output(embedder: Embedder) -> None:
    text = "Reproducibility check"
    first = await embedder.embed_one(text)
    second = await embedder.embed_one(text)
    assert first == second