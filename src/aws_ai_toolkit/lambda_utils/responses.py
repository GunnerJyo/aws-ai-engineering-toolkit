"""Minimal API Gateway response and event validation helpers."""

import json
from typing import Any


def json_response(status_code: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON HTTP API proxy response."""
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json; charset=utf-8"},
        "body": json.dumps(payload),
    }


def error_response(status_code: int, message: str) -> dict[str, Any]:
    """Build a safe client-facing error response."""
    return json_response(status_code, {"error": message})


def parse_prompt(event: dict[str, Any]) -> str:
    """Validate a direct or API Gateway proxy event and return its prompt."""
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("Request body must be an object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > 20_000:
        raise ValueError("prompt exceeds the 20000 character limit")
    return prompt
