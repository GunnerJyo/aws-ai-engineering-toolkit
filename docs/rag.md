# Retrieval-augmented generation utilities

`chunk_text` splits characters with metadata offsets and overlap. `TitanV2Embeddings` calls Bedrock Runtime `InvokeModel` for Titan V2 (256, 512, or 1024 dimensions). `Retriever` accepts any object implementing `embed(text) -> list[float]`, then indexes and retrieves with cosine similarity. The default store is deliberately in-memory and nonpersistent. Titan requests require `bedrock:InvokeModel` and may incur charges. The included RAG example shows document → chunks → vectors → search → context; it does not call a chat model.
