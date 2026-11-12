"""Wrapper around Bedrock Runtime ApplyGuardrail."""

from dataclasses import dataclass
from typing import Any, Literal

from botocore.exceptions import BotoCoreError, ClientError

from aws_ai_toolkit.bedrock.exceptions import translate_bedrock_error

GuardrailSource = Literal["INPUT", "OUTPUT"]


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Structured policy decision plus AWS assessment details."""

    action: str
    intervened: bool
    action_reason: str | None
    outputs: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    raw: dict[str, Any]


def apply_guardrail(
    client: Any, guardrail_id: str, version: str, text: str, *, source: GuardrailSource = "INPUT"
) -> GuardrailResult:
    """Evaluate text independently and preserve assessments.

    INPUT is intended for user-provided content before model processing;
    OUTPUT is intended for generated content before it is returned. If the
    evaluation request fails or its action is unrecognized, this raises and
    callers must not continue with unverified content.
    """
    if not guardrail_id.strip() or not version.strip() or not text.strip():
        raise ValueError("guardrail_id, version, and text must not be empty")
    if source not in ("INPUT", "OUTPUT"):
        raise ValueError("source must be INPUT or OUTPUT")
    try:
        response = client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=version,
            source=source,
            content=[{"text": {"text": text}}],
        )
    except (ClientError, BotoCoreError) as exc:
        raise translate_bedrock_error(exc, "ApplyGuardrail") from exc
    action = response.get("action")
    if action not in ("NONE", "GUARDRAIL_INTERVENED"):
        raise ValueError("ApplyGuardrail response is missing a recognized action")
    return GuardrailResult(
        action,
        action == "GUARDRAIL_INTERVENED",
        response.get("actionReason"),
        response.get("outputs", []),
        response.get("assessments", []),
        response,
    )
