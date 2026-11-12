"""Bedrock error types and safe translation of common service failures."""

from botocore.exceptions import BotoCoreError, ClientError


class BedrockRequestError(RuntimeError):
    """A Bedrock request failed; original AWS error is chained as the cause."""


class BedrockThrottlingError(BedrockRequestError):
    """Bedrock rejected a request because of throttling or a service quota."""


class BedrockAuthorizationError(BedrockRequestError):
    """The caller is not authorized to perform the Bedrock operation."""


_THROTTLING_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
    "ServiceQuotaExceededException",
}
_AUTHORIZATION_CODES = {"AccessDeniedException", "UnauthorizedException"}


def translate_bedrock_error(
    exc: ClientError | BotoCoreError, operation: str
) -> BedrockRequestError:
    """Create a safe typed error without exposing AWS messages or request data."""
    code = exc.response.get("Error", {}).get("Code") if isinstance(exc, ClientError) else None
    error_type = (
        BedrockThrottlingError
        if code in _THROTTLING_CODES
        else BedrockAuthorizationError
        if code in _AUTHORIZATION_CODES
        else BedrockRequestError
    )
    return error_type(f"Amazon Bedrock {operation} request failed")
