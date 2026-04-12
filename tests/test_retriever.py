from unittest.mock import patch
from core.retriever import retrieve_chunks
from core.chroma_client import get_vector_store
from core.exceptions import RetrievalError

INFLUENCER = "test_influencer"

# ── Setup ──────────────────────────────────────────────────────────────────────
print("SETUP: Storing test chunks")
store = get_vector_store(INFLUENCER)
chunks = [
    "الإنتاجية تعني إنجاز المهام المهمة أولاً",
    "لازم تصحى بدري عشان تنجز",
    "السيارة الكورية اقتصادية جداً",
    "النوم المبكر يساعد على التركيز",
    "الرياضة اليومية تحسن الإنتاجية",
]
store.add_texts(texts=chunks, ids=["1", "2", "3", "4", "5"])
print(f"  stored {len(chunks)} chunks\n")

# ── TEST 1: Basic retrieval ────────────────────────────────────────────────────
print("TEST 1: Basic retrieval returns k=4 chunks")
results = retrieve_chunks("كيف أكون منتجاً؟", INFLUENCER, k=4)
print(f"  retrieved: {len(results)} chunks")
assert len(results) == 4
print("  PASSED\n")

# ── TEST 2: Respects k parameter ──────────────────────────────────────────────
print("TEST 2: Respects k=2")
results = retrieve_chunks("كيف أكون منتجاً؟", INFLUENCER, k=2)
print(f"  retrieved: {len(results)} chunks")
assert len(results) == 2
print("  PASSED\n")

# ── TEST 3: Default k=4 ───────────────────────────────────────────────────────
print("TEST 3: Default k=4 is used when k is not specified")
results = retrieve_chunks("النوم والراحة", INFLUENCER)
print(f"  retrieved: {len(results)} chunks")
assert len(results) == 4
print("  PASSED\n")

# ── TEST 4: Returns list[str] and top result is semantically relevant ──────────
print("TEST 4: Returns plain strings and top result is semantically relevant")
results = retrieve_chunks("الإنتاجية والإنجاز", INFLUENCER, k=1)
print(f"  type: {type(results[0]).__name__}")
print(f"  top result: {results[0]}")
assert isinstance(results[0], str)
assert "الإنتاجية" in results[0]
print("  PASSED\n")

# ── TEST 5: RetrievalError on broken vector store ─────────────────────────────
print("TEST 5: Raises RetrievalError when vector store fails")
with patch("core.retriever.get_vector_store", side_effect=Exception("connection refused")):
    try:
        retrieve_chunks("test", INFLUENCER, k=2)
        print("  FAILED — should have raised RetrievalError")
    except RetrievalError as e:
        print(f"  PASSED — got RetrievalError: {e}\n")

# ── Teardown ───────────────────────────────────────────────────────────────────
store._client.delete_collection(INFLUENCER)
print("All retriever tests passed.")

