"""Botocore contract tests for Bedrock Converse; no HTTP requests are made."""

import boto3
import pytest
from botocore.stub import Stubber

from aws_ai_toolkit.bedrock import (
    BedrockAuthorizationError,
    BedrockThrottlingError,
    converse,
    create_bedrock_runtime_client,
)
from aws_ai_toolkit.config import ToolkitConfig


@pytest.fixture
def runtime_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
    )


def test_converse_request_and_mixed_response_contract(runtime_client):
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Use "},
                    {"toolUse": {"toolUseId": "call-1", "name": "lookup", "input": {"q": "x"}}},
                    {"text": "the tool."},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9},
        "metrics": {"latencyMs": 23},
    }
    expected = {
        "modelId": "example-model",
        "messages": [{"role": "user", "content": [{"text": "Find an answer."}]}],
        "system": [{"text": "Be concise."}],
        "inferenceConfig": {"maxTokens": 31, "temperature": 0.2},
    }
    with Stubber(runtime_client) as stubber:
        stubber.add_response("converse", response, expected)
        result = converse(
            runtime_client,
            "example-model",
            "Find an answer.",
            system="Be concise.",
            temperature=0.2,
            max_tokens=31,
        )
        stubber.assert_no_pending_responses()
    assert result.text == "Use the tool."
    assert result.stop_reason == "tool_use"
    assert result.usage == response["usage"]
    assert result.content_blocks == response["output"]["message"]["content"]


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("ThrottlingException", BedrockThrottlingError),
        ("AccessDeniedException", BedrockAuthorizationError),
    ],
)
def test_converse_aws_errors_are_typed_and_chained(runtime_client, code, error_type):
    with Stubber(runtime_client) as stubber:
        stubber.add_client_error("converse", service_error_code=code, http_status_code=400)
        with pytest.raises(error_type) as raised:
            converse(runtime_client, "example-model", "prompt")
    assert raised.value.__cause__ is not None
    assert "test-secret-key" not in str(raised.value)


def test_client_configuration_sets_timeouts_and_standard_retries(monkeypatch):
    captured = {}

    def fake_client(service_name, **kwargs):
        captured.update(service_name=service_name, **kwargs)
        return object()

    monkeypatch.setattr(boto3, "client", fake_client)
    create_bedrock_runtime_client(
        ToolkitConfig(
            region_name="us-west-2",
            max_attempts=5,
            connect_timeout_seconds=3,
            read_timeout_seconds=90,
        )
    )
    assert captured["service_name"] == "bedrock-runtime"
    assert captured["region_name"] == "us-west-2"
    assert captured["config"].connect_timeout == 3
    assert captured["config"].read_timeout == 90
    assert captured["config"].retries == {"mode": "standard", "total_max_attempts": 5}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connect_timeout_seconds": 0},
        {"read_timeout_seconds": float("nan")},
        {"read_timeout_seconds": True},
        {"connect_timeout_seconds": 10**1000},
        {"max_attempts": True},
    ],
)
def test_client_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ToolkitConfig(**kwargs)
