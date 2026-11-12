"""Amazon Titan Text Embeddings V2 adapter."""

import json
import math
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from aws_ai_toolkit.bedrock.exceptions import translate_bedrock_error


class Embedder(Protocol):
    """Minimal interface accepted by the retriever."""

    def embed(self, text: str) -> list[float]: ...


class TitanV2Embeddings:
    """Create Titan V2 float embeddings using Bedrock Runtime.

    Titan V2 accepts at most 50,000 characters or 8,192 tokens. This adapter
    enforces the character limit; exact token counting requires model-aware
    tokenization and is intentionally not approximated here.
    """

    SUPPORTED_DIMENSIONS = {256, 512, 1024}
    MAX_INPUT_CHARACTERS = 50_000

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def __init__(self, client: Any, dimensions: int = 1024, *, normalize: bool = True) -> None:
        if dimensions not in self.SUPPORTED_DIMENSIONS:
            raise ValueError("dimensions must be one of 256, 512, or 1024")
        self.client = client
        self.dimensions = dimensions
        self.normalize = normalize

    def embed(self, text: str) -> list[float]:
        """Embed non-empty text, validating the returned vector."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty")
        if len(text) > self.MAX_INPUT_CHARACTERS:
            raise ValueError("text exceeds Titan Text Embeddings V2's 50000 character limit")
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
            raise translate_bedrock_error(exc, "Titan embedding") from exc
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, AttributeError) as exc:
            raise ValueError(
                "Invalid Titan embedding response: body must contain valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("Invalid Titan embedding response: JSON root must be an object")
        vector = payload.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != self.dimensions
            or any(not self._is_finite_number(value) for value in vector)
        ):
            raise ValueError(
                "Invalid Titan embedding response: expected "
                f"{self.dimensions} finite numeric values"
            )
        return [float(value) for value in vector]
