"""Validated configuration for AWS clients."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolkitConfig:
    """Settings shared by toolkit components; credentials use the boto3 provider chain."""

    region_name: str | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.region_name is not None and not self.region_name.strip():
            raise ValueError("region_name must not be empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
