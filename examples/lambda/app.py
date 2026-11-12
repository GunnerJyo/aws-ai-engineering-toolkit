"""Minimal API Gateway Lambda handler using its execution role."""

import json
import os

import boto3


def handler(event, context):  # noqa: ARG001
    """Validate an API event, invoke Bedrock, and return a safe HTTP response."""
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return response(400, {"error": "Request body must be valid JSON"})
    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20_000:
        return response(400, {"error": "prompt must be a non-empty string up to 20000 characters"})
    try:
        runtime = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
        result = runtime.converse(
            modelId=os.environ["BEDROCK_MODEL_ID"],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": int(os.getenv("MAX_TOKENS", "512"))},
        )
        text = "".join(
            block.get("text", "")
            for block in result["output"]["message"]["content"]
            if "text" in block
        )
        return response(200, {"text": text})
    except Exception:  # AWS exceptions may include request internals; return no details.
        return response(502, {"error": "Model request failed"})


def response(status, payload):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json; charset=utf-8"},
        "body": json.dumps(payload),
    }
