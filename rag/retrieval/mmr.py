"""Maximal Marginal Relevance (MMR) selection.

MMR iteratively selects items that are relevant to the query while penalizing
redundancy with already-selected items, trading off relevance vs. diversity:

    score(c) = λ · sim(query, c) − (1 − λ) · max_{s ∈ selected} sim(c, s)

Dense embeddings here are L2-normalized (the embedder normalizes), so cosine
similarity reduces to a dot product.
"""

import numpy as np


def mmr_select(
    query_vec: list[float],
    candidate_vecs: list[list[float]],
    k: int,
    lambda_mult: float = 0.6,
) -> list[int]:
    """Return indices of the selected candidates, ordered by selection.

    Args:
        query_vec: query embedding (normalized).
        candidate_vecs: candidate embeddings (normalized), aligned by index.
        k: number of items to select.
        lambda_mult: relevance weight in [0, 1]; higher favors relevance.
    """
    n = len(candidate_vecs)
    if n == 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))

    query = np.asarray(query_vec, dtype=np.float32)
    cands = np.asarray(candidate_vecs, dtype=np.float32)

    # Relevance of each candidate to the query (cosine ≈ dot for normalized vectors).
    relevance = cands @ query
    # Pairwise candidate-candidate similarity for the diversity penalty.
    pairwise = cands @ cands.T

    selected: list[int] = []
    remaining = set(range(n))

    # Seed with the most relevant candidate.
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.discard(first)

    while len(selected) < k and remaining:
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            redundancy = max(pairwise[idx, s] for s in selected)
            score = lambda_mult * relevance[idx] - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected
