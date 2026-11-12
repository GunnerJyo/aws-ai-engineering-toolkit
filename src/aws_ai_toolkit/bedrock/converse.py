"""Model-agnostic Converse API helper."""

from dataclasses import dataclass
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from .exceptions import BedrockRequestError


@dataclass(frozen=True, slots=True)
class ConverseResponse:
    """Stable subset of a Converse response."""

    text: str
    stop_reason: str | None
    usage: dict[str, int]
    raw: dict[str, Any]


def converse(
    client: Any,
    model_id: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 512,
) -> ConverseResponse:
    """Send one user prompt through Converse and return concatenated text blocks."""
    if not model_id.strip() or not prompt.strip():
        raise ValueError("model_id and prompt must not be empty")
    if max_tokens < 1 or (temperature is not None and not 0 <= temperature <= 1):
        raise ValueError("max_tokens must be positive and temperature must be between 0 and 1")
    request: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if system:
        request["system"] = [{"text": system}]
    if temperature is not None:
        request["inferenceConfig"]["temperature"] = temperature
    try:
        result = client.converse(**request)
    except (ClientError, BotoCoreError) as exc:
        raise BedrockRequestError("Amazon Bedrock Converse request failed") from exc
    message = result.get("output", {}).get("message", {})
    content = message.get("content", [])
    text = "".join(block.get("text", "") for block in content if "text" in block)
    return ConverseResponse(text, result.get("stopReason"), result.get("usage", {}), result)
