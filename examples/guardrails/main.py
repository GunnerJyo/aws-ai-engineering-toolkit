"""Evaluate both input and generated text independently of model invocation."""

import os

from aws_ai_toolkit.bedrock import create_bedrock_runtime_client
from aws_ai_toolkit.config import ToolkitConfig
from aws_ai_toolkit.guardrails import apply_guardrail

client = create_bedrock_runtime_client(ToolkitConfig(region_name=os.getenv("AWS_REGION")))
guardrail_id = os.environ["BEDROCK_GUARDRAIL_ID"]
version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

# ApplyGuardrail calls the guardrail service only; it does not invoke a foundation model.
input_check = apply_guardrail(client, guardrail_id, version, "User prompt", source="INPUT")
if not input_check.intervened:
    generated_text = "A sample generated response."
    output_check = apply_guardrail(client, guardrail_id, version, generated_text, source="OUTPUT")
    if not output_check.intervened:
        print(generated_text)
