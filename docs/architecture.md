# Architecture

The `aws_ai_toolkit` package contains independent Bedrock client/converse, RAG, guardrail, and Lambda response utilities. AWS calls accept injected boto3-like clients so unit tests can use stubs. The RAG retriever accepts an embedder protocol and a replaceable in-memory store. Credentials always come from the standard AWS provider chain. The Strands example is optional and lives outside the core package.

The store is in-process memory only: it has no persistence, metadata filtering, access control, or scaling guarantees.
