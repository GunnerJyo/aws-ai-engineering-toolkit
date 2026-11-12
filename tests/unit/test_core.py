"""Offline tests for core APIs; no credentials or network access are used."""

import io
import json

import pytest
from botocore.exceptions import ClientError

from aws_ai_toolkit.bedrock import BedrockRequestError, converse
from aws_ai_toolkit.config import ToolkitConfig
from aws_ai_toolkit.guardrails import apply_guardrail
from aws_ai_toolkit.lambda_utils import json_response, parse_prompt
from aws_ai_toolkit.rag import Document, InMemoryVectorStore, TitanV2Embeddings, chunk_text
from aws_ai_toolkit.rag.vector_store import cosine_similarity


class Stub:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.request = result or {}, error, None

    def converse(self, **request):
        self.request = request
        if self.error:
            raise self.error
        return self.result

    def apply_guardrail(self, **request):
        self.request = request
        if self.error:
            raise self.error
        return self.result

    def invoke_model(self, **request):
        self.request = request
        return {"body": io.BytesIO(json.dumps({"embedding": [0.1] * 256}).encode())}


def test_config_validation():
    assert ToolkitConfig("us-east-1").region_name == "us-east-1"
    with pytest.raises(ValueError):
        ToolkitConfig(max_attempts=0)


def test_converse_formats_request_and_extracts_response():
    stub = Stub({"output": {"message": {"content": [{"text": "hello"}]}}, "usage": {"x": 1}})
    result = converse(stub, "model", "prompt", system="instruction", temperature=0.2)
    assert result.text == "hello"
    assert stub.request["system"] == [{"text": "instruction"}]
    assert stub.request["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.2}


def test_converse_wraps_aws_errors():
    error = ClientError({"Error": {"Code": "Denied", "Message": "no"}}, "Converse")
    with pytest.raises(BedrockRequestError) as raised:
        converse(Stub(error=error), "model", "prompt")
    assert raised.value.__cause__ is error


def test_chunking_preserves_metadata_and_validates():
    chunks = chunk_text("abcdefgh", chunk_size=4, overlap=1, metadata={"source": "x"})
    assert [item.text for item in chunks] == ["abcd", "defg", "gh"]
    assert chunks[0].metadata["source"] == "x"
    assert chunk_text("  ") == []
    with pytest.raises(ValueError):
        chunk_text("a", chunk_size=2, overlap=2)


def test_embedding_validation():
    stub = Stub()
    assert len(TitanV2Embeddings(stub, 256).embed("hi")) == 256
    assert json.loads(stub.request["body"])["dimensions"] == 256
    with pytest.raises(ValueError):
        TitanV2Embeddings(stub, 128)


def test_vector_store_similarity_dimension_and_top_k():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    store = InMemoryVectorStore()
    assert store.search([1, 0]) == []
    store.add(Document("best"), [1, 0])
    store.add(Document("other"), [0, 1])
    assert store.search([1, 0], top_k=1)[0].document.text == "best"
    with pytest.raises(ValueError):
        store.add(Document("bad"), [1])


def test_guardrail_response_and_errors():
    result = apply_guardrail(
        Stub({"action": "GUARDRAIL_INTERVENED", "assessments": [{"x": 1}]}),
        "abc",
        "DRAFT",
        "text",
        source="OUTPUT",
    )
    assert result.intervened and result.assessments == [{"x": 1}]
    error = ClientError({"Error": {"Code": "Denied", "Message": "no"}}, "ApplyGuardrail")
    with pytest.raises(BedrockRequestError):
        apply_guardrail(Stub(error=error), "abc", "1", "text")


def test_lambda_input_and_response():
    assert parse_prompt({"body": '{"prompt":"hello"}'}) == "hello"
    with pytest.raises(ValueError):
        parse_prompt({"body": {"prompt": " "}})
    assert json_response(200, {"ok": True})["statusCode"] == 200
