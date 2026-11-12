"""In-memory retrieval workflow."""

from .chunking import chunk_text
from .embeddings import Embedder
from .vector_store import InMemoryVectorStore, SearchResult


class Retriever:
    """Connect chunking, any compatible embedder, and an in-memory vector store."""

    def __init__(self, embedder: Embedder, store: InMemoryVectorStore | None = None) -> None:
        self.embedder = embedder
        self.store = store or InMemoryVectorStore()

    def add_text(
        self, text: str, *, chunk_size: int = 1000, overlap: int = 100, metadata: dict | None = None
    ) -> int:
        """Chunk, embed, and index text; returns the number of indexed chunks."""
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, metadata=metadata)
        for document in chunks:
            self.store.add(document, self.embedder.embed(document.text))
        return len(chunks)

    def retrieve(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        """Retrieve relevant chunks for a non-empty query."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.store.search(self.embedder.embed(query), top_k=top_k)

    def context(self, query: str, *, top_k: int = 3) -> str:
        """Format retrieved document text as a compact context string."""
        return "\n\n".join(result.document.text for result in self.retrieve(query, top_k=top_k))
