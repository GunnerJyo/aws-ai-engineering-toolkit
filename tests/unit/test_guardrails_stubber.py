"""Botocore contract tests for independent ApplyGuardrail calls."""

import boto3
import pytest
from botocore.stub import Stubber

from aws_ai_toolkit.bedrock import BedrockAuthorizationError, BedrockRequestError
from aws_ai_toolkit.guardrails import apply_guardrail


@pytest.fixture
def runtime_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
    )


@pytest.mark.parametrize("source", ["INPUT", "OUTPUT"])
def test_apply_guardrail_request_and_response_contract(runtime_client, source):
    response = {
        "usage": {
            "topicPolicyUnits": 0,
            "contentPolicyUnits": 1,
            "wordPolicyUnits": 0,
            "sensitiveInformationPolicyUnits": 0,
            "sensitiveInformationPolicyFreeUnits": 0,
            "contextualGroundingPolicyUnits": 0,
        },
        "action": "NONE",
        "outputs": [],
        "assessments": [{"contentPolicy": {"filters": []}}],
    }
    expected = {
        "guardrailIdentifier": "abc123",
        "guardrailVersion": "1",
        "source": source,
        "content": [{"text": {"text": "safe sample"}}],
    }
    with Stubber(runtime_client) as stubber:
        stubber.add_response("apply_guardrail", response, expected)
        result = apply_guardrail(runtime_client, "abc123", "1", "safe sample", source=source)
        stubber.assert_no_pending_responses()
    assert result.action == "NONE"
    assert result.intervened is False
    assert result.assessments == response["assessments"]


def test_apply_guardrail_aws_failures_are_never_approved(runtime_client):
    with Stubber(runtime_client) as stubber:
        stubber.add_client_error(
            "apply_guardrail", service_error_code="AccessDeniedException", http_status_code=403
        )
        with pytest.raises(BedrockAuthorizationError) as raised:
            apply_guardrail(runtime_client, "abc123", "1", "unverified content")
    assert isinstance(raised.value, BedrockRequestError)
    assert raised.value.__cause__ is not None


@pytest.mark.parametrize("response", [{}, {"action": "UNKNOWN"}])
def test_missing_or_unexpected_actions_raise(response):
    class ResponseClient:
        def apply_guardrail(self, **kwargs):
            return response

    with pytest.raises(ValueError, match="recognized action"):
        apply_guardrail(ResponseClient(), "abc123", "1", "unverified content")
