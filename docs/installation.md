# Installation

Requires Python 3.11+. From the repository root, use `python -m pip install -e ".[dev]"` for development or `python -m pip install .` for core runtime use. Add the optional Strands integration with `python -m pip install -e ".[strands]"`. Configure AWS credentials using an AWS-supported boto3 provider and set `AWS_REGION`; do not place credentials in source files.
