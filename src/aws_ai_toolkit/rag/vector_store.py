"""Small dependency-free in-memory cosine vector store."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .chunking import Document


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, rejecting invalid dimensions and zero vectors."""
    if not left or len(left) != len(right):
        raise ValueError("vectors must have the same non-zero dimension")
    if any(not _is_finite_number(v) for v in (*left, *right)):
        raise ValueError("vectors must contain only finite values")
    denominator = math.sqrt(sum(v * v for v in left) * sum(v * v for v in right))
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


@dataclass(frozen=True, slots=True)
class SearchResult:
    document: Document
    score: float


class InMemoryVectorStore:
    """In-memory cosine store; ties retain document insertion order.

    Contents are not persistent. Python's stable sorting means equal scores
    retain the order in which documents were added.
    """

    def __init__(self) -> None:
        self._items: list[tuple[tuple[float, ...], Document]] = []
        self._dimension: int | None = None

    def add(self, document: Document, vector: Sequence[float]) -> None:
        """Add a document and its embedding after validating dimension and values."""
        if not vector or any(not _is_finite_number(value) for value in vector):
            raise ValueError("vector must be non-empty and finite")
        if self._dimension is not None and self._dimension != len(vector):
            raise ValueError("vector dimension mismatch")
        self._dimension = len(vector)
        self._items.append((tuple(float(v) for v in vector), document))

    def search(self, query: Sequence[float], *, top_k: int = 3) -> list[SearchResult]:
        """Return top_k results by descending score, breaking ties by insertion order."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._items:
            return []
        if len(query) != self._dimension:
            raise ValueError("query vector dimension mismatch")
        ranked = [
            SearchResult(doc, cosine_similarity(query, vector)) for vector, doc in self._items
        ]
        return sorted(ranked, key=lambda result: result.score, reverse=True)[:top_k]
