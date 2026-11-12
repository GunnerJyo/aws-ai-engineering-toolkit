"""RAG flow example using Titan V2; invokes paid Bedrock embedding requests."""

import os

from aws_ai_toolkit.bedrock import create_bedrock_runtime_client
from aws_ai_toolkit.config import ToolkitConfig
from aws_ai_toolkit.rag import Retriever, TitanV2Embeddings

runtime = create_bedrock_runtime_client(ToolkitConfig(region_name=os.getenv("AWS_REGION")))
retriever = Retriever(TitanV2Embeddings(runtime, dimensions=256))
retriever.add_text(
    "Amazon Bedrock is a managed service for building generative AI applications. "
    "Titan Text Embeddings V2 converts text into vectors for semantic search.",
    chunk_size=90,
    overlap=10,
    metadata={"source": "example"},
)
print(retriever.context("How do embeddings help search?", top_k=2))
