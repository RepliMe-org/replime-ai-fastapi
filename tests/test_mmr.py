"""Unit tests for rag/retrieval/mmr.py — Maximal Marginal Relevance selection."""
from rag.retrieval.mmr import mmr_select


def test_empty_candidates():
    assert mmr_select([1.0, 0.0], [], k=3) == []


def test_k_zero_returns_empty():
    assert mmr_select([1.0, 0.0], [[1.0, 0.0]], k=0) == []


def test_fewer_candidates_than_k_returns_all():
    assert mmr_select([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], k=5) == [0, 1]


def test_first_pick_is_most_relevant():
    sel = mmr_select([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]], k=1, lambda_mult=0.6)
    assert sel == [1]


def test_high_lambda_favors_relevance():
    # Two identical highly-relevant vectors + one diverse one.
    q = [1.0, 0.0]
    cands = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert mmr_select(q, cands, k=2, lambda_mult=0.6) == [0, 1]


def test_low_lambda_favors_diversity():
    q = [1.0, 0.0]
    cands = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert mmr_select(q, cands, k=2, lambda_mult=0.3) == [0, 2]


def test_selection_is_unique_and_sized():
    q = [1.0, 0.0]
    cands = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    sel = mmr_select(q, cands, k=3, lambda_mult=0.6)
    assert len(sel) == 3
    assert len(set(sel)) == 3
