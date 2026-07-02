"""Component-level micro-benchmarks for the CPU-bound stages of the pipeline.

These measure only the local, deterministic components — language detection,
Arabic normalization, chunking, MMR reranking, citation extraction, and prompt
assembly. They involve no network, no embedding model, and no LLM, so the
numbers are reproducible on any machine and cost nothing to run.

End-to-end latency (embedding + Qdrant retrieval + LLM generation) is dominated
by network/model/provider time and is measured separately with Locust against a
running server — see load_tests/locustfile.py.

Run:  python benchmarks/component_benchmarks.py
"""
import random
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.llm.prompt_builder import build_messages  # noqa: E402
from rag.retrieval.mmr import mmr_select  # noqa: E402
from rag.text.chunker import chunk_transcript  # noqa: E402
from rag.text.language_detector import detect_language  # noqa: E402
from rag.text.text_normalizer import normalize_arabic  # noqa: E402
from schemas.chat import ChatbotConfig  # noqa: E402
from services.chat_service import _extract_cited_chunks  # noqa: E402


def _bench(name: str, fn, iterations: int = 2000) -> None:
    for _ in range(min(50, iterations)):  # warmup
        fn()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    samples.sort()
    mean = statistics.mean(samples)
    p95 = samples[int(len(samples) * 0.95)]
    print(f"{name:<34} iters={iterations:>5}  mean={mean:8.4f} ms  p95={p95:8.4f} ms  ops/s={1000 / mean:>10,.0f}")


def main() -> None:
    en_text = "What is the difference between supervised and unsupervised learning in AI?"
    ar_text = "ما هو الفرق بين التعلم الخاضع للإشراف والتعلم غير الخاضع للإشراف في الذكاء الاصطناعي؟"
    ar_dirty = "الأمثلة المُهِمّة في تعلُّم الآلة والذكاء الاصطناعي " * 5
    segments = [
        {"text": f"This is transcript segment number {i} discussing machine learning concepts.", "start": i * 3}
        for i in range(60)
    ]
    query_vec = [0.1] * 1024  # multilingual-e5-large dimension
    random.seed(0)
    candidate_vecs = [[random.random() for _ in range(1024)] for _ in range(20)]
    chunks = [
        {
            "chunk_text": f"chunk {i} content",
            "youtube_video_id": f"v{i}",
            "video_title": f"Video {i}",
            "timestamp_seconds": i * 10,
            "similarity_score": 0.9 - i * 0.01,
        }
        for i in range(5)
    ]
    answer = "First point [1]. Second point [2, 3]. Third point [4]. Fourth point [5]."
    config = ChatbotConfig(
        chatbot_name="Bench", talk_like_me=False, tone="FRIENDLY",
        verbosity="BALANCED", formality="NEUTRAL",
    )

    print("Component micro-benchmarks — CPU-bound stages only (no network/model/LLM)")
    print("-" * 100)
    _bench("language_detect (English)", lambda: detect_language(en_text))
    _bench("language_detect (Arabic)", lambda: detect_language(ar_text))
    _bench("normalize_arabic", lambda: normalize_arabic(ar_dirty))
    _bench("chunk_transcript (60 segments)", lambda: chunk_transcript(segments), iterations=500)
    _bench("mmr_select (20 -> 5, dim 1024)", lambda: mmr_select(query_vec, candidate_vecs, k=5), iterations=1000)
    _bench("extract_citations", lambda: _extract_cited_chunks(answer, chunks))
    _bench("build_messages (5 chunks)", lambda: build_messages("question", chunks, [], config, "en"))
    print("-" * 100)


if __name__ == "__main__":
    main()
