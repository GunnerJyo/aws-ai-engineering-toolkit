"""Character-based document chunking that preserves source metadata."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """Text and caller-supplied metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str, *, chunk_size: int = 1000, overlap: int = 100, metadata: dict[str, Any] | None = None
) -> list[Document]:
    """Split text by characters; overlap must be smaller than chunk_size."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text.strip():
        return []
    chunks: list[Document] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(Document(chunk, {**(metadata or {}), "start": start, "end": end}))
        start += step
    return chunks
