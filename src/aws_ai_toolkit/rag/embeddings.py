"""Amazon Titan Text Embeddings V2 adapter."""

import json
import math
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from aws_ai_toolkit.bedrock.exceptions import BedrockRequestError


class Embedder(Protocol):
    """Minimal interface accepted by the retriever."""

    def embed(self, text: str) -> list[float]: ...


class TitanV2Embeddings:
    """Create normalized Titan V2 float embeddings using Bedrock Runtime."""

    SUPPORTED_DIMENSIONS = {256, 512, 1024}

    def __init__(self, client: Any, dimensions: int = 1024, *, normalize: bool = True) -> None:
        if dimensions not in self.SUPPORTED_DIMENSIONS:
            raise ValueError("dimensions must be one of 256, 512, or 1024")
        self.client = client
        self.dimensions = dimensions
        self.normalize = normalize

    def embed(self, text: str) -> list[float]:
        """Embed non-empty text, validating the returned vector."""
        if not text.strip():
            raise ValueError("text must not be empty")
        try:
            response = self.client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps(
                    {"inputText": text, "dimensions": self.dimensions, "normalize": self.normalize}
                ),
                accept="application/json",
                contentType="application/json",
            )
            payload = json.loads(response["body"].read())
        except (ClientError, BotoCoreError) as exc:
            raise BedrockRequestError("Titan embedding request failed") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid Titan embedding response") from exc
        vector = payload.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != self.dimensions
            or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in vector)
        ):
            raise ValueError("Embedding response has an invalid vector")
        return [float(value) for value in vector]
