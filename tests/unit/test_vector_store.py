"""Edge-case and deterministic ranking tests for the in-memory vector store."""

import math

import pytest

from aws_ai_toolkit.rag import Document, InMemoryVectorStore
from aws_ai_toolkit.rag.vector_store import cosine_similarity


def test_empty_store_and_top_k_boundaries():
    store = InMemoryVectorStore()
    assert store.search([1.0, 0.0]) == []
    store.add(Document("a"), [1.0, 0.0])
    assert [hit.document.text for hit in store.search([1.0, 0.0], top_k=1)] == ["a"]
    assert len(store.search([1.0, 0.0], top_k=100)) == 1
    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0], top_k=0)


def test_zero_vectors_dimension_mismatches_and_non_finite_values():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0], [1.0, 0.0])
    store = InMemoryVectorStore()
    store.add(Document("x"), [1.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        store.search([0.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0])
    for invalid in (math.nan, math.inf, -math.inf, 10**1000):
        with pytest.raises(ValueError, match="finite"):
            InMemoryVectorStore().add(Document("invalid"), [invalid])
        with pytest.raises(ValueError, match="finite"):
            cosine_similarity([1.0], [invalid])


def test_duplicate_scores_keep_insertion_order_at_top_k_boundary():
    store = InMemoryVectorStore()
    for label in ("first", "second", "third"):
        store.add(Document(label), [1.0, 0.0])
    assert [hit.document.text for hit in store.search([1.0, 0.0], top_k=2)] == [
        "first",
        "second",
    ]
