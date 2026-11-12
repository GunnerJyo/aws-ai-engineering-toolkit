"""Bedrock error types."""


class BedrockRequestError(RuntimeError):
    """A Bedrock request failed; original AWS error is chained as the cause."""
