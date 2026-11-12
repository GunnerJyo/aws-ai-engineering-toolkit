"""Tiny deterministic retrieval dataset; this is a mechanics regression set, not a benchmark."""

from collections.abc import Sequence
from dataclasses import dataclass

from aws_ai_toolkit.rag import Document, InMemoryVectorStore


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    vector: tuple[float, ...]
    relevant_ids: frozenset[str]


DOCUMENTS = (
    (
        "bedrock",
        "Bedrock Converse sends text messages to a foundation model.",
        (1.0, 0.0, 0.0, 0.0),
    ),
    (
        "titan",
        "Titan Text Embeddings V2 represents text as vectors for semantic search.",
        (0.0, 0.0, 1.0, 0.0),
    ),
    ("lambda", "AWS Lambda runs Python handlers in response to events.", (0.0, 1.0, 0.0, 0.0)),
    (
        "cosine",
        "Cosine similarity ranks embedding vectors by their direction.",
        (0.0, 0.0, 0.8, 0.2),
    ),
)

EXAMPLES = (
    EvaluationExample(
        "How does Bedrock send a prompt to a model?", (1.0, 0.0, 0.0, 0.0), frozenset({"bedrock"})
    ),
    EvaluationExample(
        "Which service runs Python for an event?", (0.0, 1.0, 0.0, 0.0), frozenset({"lambda"})
    ),
    EvaluationExample(
        "What creates vectors for text search?", (0.0, 0.0, 1.0, 0.0), frozenset({"titan"})
    ),
)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Calculate relevant retrieved IDs divided by all relevant IDs in the top k."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def evaluate_recall_at_k(k: int = 1) -> float:
    """Return macro-average Recall@K for the fixed-vector sample corpus."""
    store = InMemoryVectorStore()
    for doc_id, text, vector in DOCUMENTS:
        store.add(Document(text, {"id": doc_id}), vector)
    scores = []
    for example in EXAMPLES:
        retrieved = store.search(example.vector, top_k=k)
        ids = [result.document.metadata["id"] for result in retrieved]
        scores.append(recall_at_k(ids, example.relevant_ids, k))
    return sum(scores) / len(scores)
