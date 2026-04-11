from core.chroma_client import get_vector_store

COLLECTION = "test_collection"

# TEST 1: get vector store
print("TEST 1: Get vector store")
vector_store = get_vector_store(COLLECTION)
print(f"  vector store: {vector_store}")
print("  PASSED\n")

# TEST 2: store chunks (embedding handled automatically)
print("TEST 2: Store chunks in ChromaDB")
chunks = [
    "الإنتاجية تعني إنجاز المهام المهمة أولاً",
    "لازم تصحى بدري عشان تنجز",
    "السيارة الكورية اقتصادية جداً",
]
ids = vector_store.add_texts(texts=chunks, ids=["1", "2", "3"])
print(f"  stored ids: {ids}")
assert len(ids) == 3
print("  PASSED\n")

# TEST 3: similarity search
print("TEST 3: Similarity search")
results = vector_store.similarity_search("كيف أكون منتجاً؟", k=2)
print(f"  top result: {results[0].page_content}")
print(f"  total results: {len(results)}")
assert len(results) == 2
print("  PASSED\n")

# TEST 4: similarity search with score
print("TEST 4: Similarity search with scores")
results_with_scores = vector_store.similarity_search_with_score("كيف أكون منتجاً؟", k=2)
for doc, score in results_with_scores:
    print(f"  score={score:.4f} | {doc.page_content}")
assert len(results_with_scores) == 2
print("  PASSED\n")

# cleanup
vector_store._client.delete_collection(COLLECTION)
print("All vector store tests passed.")
