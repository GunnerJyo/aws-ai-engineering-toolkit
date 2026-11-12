"""Explicitly opt-in live AWS integration checks; these may incur charges."""

import os

import boto3
import pytest

from aws_ai_toolkit.bedrock import converse
from aws_ai_toolkit.guardrails import apply_guardrail
from aws_ai_toolkit.rag import TitanV2Embeddings

pytestmark = pytest.mark.integration
_ENABLED = os.getenv("AWS_AI_TOOLKIT_RUN_INTEGRATION", "").lower() in {"1", "true", "yes"}


@pytest.fixture(scope="module")
def runtime_client():
    if not _ENABLED:
        pytest.skip("set AWS_AI_TOOLKIT_RUN_INTEGRATION=1 to enable live AWS checks")
    session = boto3.Session()
    if session.region_name is None:
        pytest.fail(
            "Live tests require AWS_REGION, AWS_DEFAULT_REGION, or a configured profile region"
        )
    if session.get_credentials() is None:
        pytest.fail("Live tests require AWS credentials from the standard boto3 provider chain")
    return session.client("bedrock-runtime")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"Live tests require {name} to be configured")
    return value


def test_live_converse(runtime_client):
    result = converse(
        runtime_client,
        _required_env("BEDROCK_INTEGRATION_MODEL_ID"),
        "Reply with the word ready.",
        max_tokens=16,
    )
    assert isinstance(result.text, str)
    assert result.stop_reason is not None


def test_live_titan_embeddings_v2(runtime_client):
    vector = TitanV2Embeddings(runtime_client, dimensions=256).embed(
        "Synthetic integration-test text; contains no customer data."
    )
    assert len(vector) == 256


def test_live_apply_guardrail(runtime_client):
    result = apply_guardrail(
        runtime_client,
        _required_env("BEDROCK_INTEGRATION_GUARDRAIL_ID"),
        os.getenv("BEDROCK_INTEGRATION_GUARDRAIL_VERSION", "DRAFT"),
        "Synthetic integration-test text; contains no customer data.",
        source="INPUT",
    )
    assert result.action in {"NONE", "GUARDRAIL_INTERVENED"}
