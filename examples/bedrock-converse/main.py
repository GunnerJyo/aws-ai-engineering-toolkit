"""Run one Amazon Bedrock Converse request."""

import os

from aws_ai_toolkit.bedrock import converse, create_bedrock_runtime_client
from aws_ai_toolkit.config import ToolkitConfig

client = create_bedrock_runtime_client(ToolkitConfig(region_name=os.getenv("AWS_REGION")))
result = converse(
    client,
    os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0"),
    "Explain retrieval-augmented generation in one sentence.",
    system="You are a concise AWS AI engineering tutor.",
    max_tokens=120,
)
print(result.text)
