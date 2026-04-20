import pytest

from rag.embedder import Embedder

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder(MODEL)


def test_embedding_dimension(embedder: Embedder) -> None:
    vector = embedder.embed_one("Hello, world!")
    assert len(vector) == EXPECTED_DIM


def test_same_input_same_output(embedder: Embedder) -> None:
    text = "Reproducibility check"
    assert embedder.embed_one(text) == embedder.embed_one(text)
