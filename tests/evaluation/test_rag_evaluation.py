"""Regression checks for retrieval ranking and the Recall@K evaluator."""

import pytest

from tests.evaluation.rag_eval import evaluate_recall_at_k, recall_at_k


def test_recall_at_k_counts_relevant_items_with_multiple_relevant_ids():
    assert recall_at_k(["a", "b", "c"], frozenset({"a", "c"}), 2) == 0.5


def test_synthetic_retrieval_regression_has_full_recall_at_one():
    assert evaluate_recall_at_k(1) == pytest.approx(1.0)
    assert evaluate_recall_at_k(4) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        recall_at_k(["a"], frozenset({"a"}), k)


def test_recall_rejects_empty_relevance_set():
    with pytest.raises(ValueError, match="relevant_ids"):
        recall_at_k(["a"], frozenset(), 1)
