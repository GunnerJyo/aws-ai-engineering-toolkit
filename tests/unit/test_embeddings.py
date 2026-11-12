"""Offline Titan request and opaque InvokeModel payload validation tests."""

import io
import json

import pytest
from botocore.exceptions import ClientError

from aws_ai_toolkit.bedrock import BedrockThrottlingError
from aws_ai_toolkit.rag import TitanV2Embeddings


class InvokeModelStub:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def invoke_model(self, **request):
        self.request = request
        raw = self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()
        return {"body": io.BytesIO(raw)}


def test_titan_request_body_and_response_payload_are_model_validated():
    payload = {"embedding": [0.25] * 256}
    client = InvokeModelStub(payload)
    result = TitanV2Embeddings(client, dimensions=256, normalize=False).embed("search this")
    assert result == [0.25] * 256
    assert client.request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert client.request["contentType"] == "application/json"
    assert client.request["accept"] == "application/json"
    assert json.loads(client.request["body"]) == {
        "inputText": "search this",
        "dimensions": 256,
        "normalize": False,
    }


def test_titan_character_limit_accepts_boundary_and_rejects_overflow():
    client = InvokeModelStub({"embedding": [0.1] * 256})
    embedder = TitanV2Embeddings(client, dimensions=256)
    assert len(embedder.embed("x" * 50_000)) == 256
    calls_before = client.request
    with pytest.raises(ValueError, match="50000 character limit"):
        embedder.embed("x" * 50_001)
    assert client.request is calls_before


@pytest.mark.parametrize(
    "raw_payload",
    [
        b"{not json",
        b"[]",
        json.dumps({"embedding": [True] * 256}).encode(),
        json.dumps({"embedding": [float("nan")] * 256}).encode(),
        json.dumps({"embedding": [float("inf")] * 256}).encode(),
        json.dumps({"embedding": [10**1000] * 256}).encode(),
        json.dumps({"embedding": [0.1] * 255}).encode(),
    ],
)
def test_titan_rejects_malformed_root_and_vectors(raw_payload):
    with pytest.raises(ValueError, match="Invalid Titan embedding response"):
        TitanV2Embeddings(InvokeModelStub(raw_payload), dimensions=256).embed("query")


def test_titan_wraps_service_errors_with_original_cause():
    class FailingClient:
        def invoke_model(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "busy"}}, "InvokeModel"
            )

    with pytest.raises(BedrockThrottlingError) as raised:
        TitanV2Embeddings(FailingClient()).embed("query")
    assert raised.value.__cause__ is not None
