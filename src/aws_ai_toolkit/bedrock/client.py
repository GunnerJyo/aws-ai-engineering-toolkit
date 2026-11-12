"""Boto3 Bedrock Runtime client factory."""

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from aws_ai_toolkit.config import ToolkitConfig


def create_bedrock_runtime_client(config: ToolkitConfig | None = None) -> BaseClient:
    """Create a runtime client using the standard boto3 credential chain."""
    settings = config or ToolkitConfig()
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.region_name,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": settings.max_attempts},
            connect_timeout=settings.connect_timeout_seconds,
            read_timeout=settings.read_timeout_seconds,
        ),
    )
