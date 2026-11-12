"""Validated configuration for AWS clients."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolkitConfig:
    """Settings shared by toolkit components; credentials use the boto3 provider chain."""

    region_name: str | None = None
    max_attempts: int = 3
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.region_name is not None and (
            not isinstance(self.region_name, str) or not self.region_name.strip()
        ):
            raise ValueError("region_name must not be empty")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be at least 1")
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("read_timeout_seconds", self.read_timeout_seconds),
        ):
            try:
                valid = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite positive number")
