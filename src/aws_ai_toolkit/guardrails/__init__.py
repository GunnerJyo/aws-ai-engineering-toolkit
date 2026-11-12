"""Independent Amazon Bedrock Guardrails evaluation."""

from .client import GuardrailResult, apply_guardrail

__all__ = ["GuardrailResult", "apply_guardrail"]
