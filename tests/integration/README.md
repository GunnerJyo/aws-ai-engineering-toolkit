# Optional live AWS integration checks

The three integration tests cover Bedrock Converse, Titan Text Embeddings V2, and independent ApplyGuardrail. Pytest collects them but skips them by default. They are not enabled in ordinary CI.

Live checks make real AWS requests and may incur model or Guardrails charges. They require configured credentials and region from the standard boto3 provider chain and must use synthetic text only. Never use production customer data.

Configure `AWS_AI_TOOLKIT_RUN_INTEGRATION=1`, `AWS_REGION` (or `AWS_DEFAULT_REGION`/profile region), and `BEDROCK_INTEGRATION_MODEL_ID`. For the guardrail test also set `BEDROCK_INTEGRATION_GUARDRAIL_ID` and optionally `BEDROCK_INTEGRATION_GUARDRAIL_VERSION` (defaults to `DRAFT`). The credentials need scoped `bedrock:InvokeModel` and/or `bedrock:ApplyGuardrail` permissions for those resources.

```bash
AWS_AI_TOOLKIT_RUN_INTEGRATION=1 \
AWS_REGION=us-east-1 \
BEDROCK_INTEGRATION_MODEL_ID=your-enabled-model-id \
BEDROCK_INTEGRATION_GUARDRAIL_ID=your-guardrail-id \
python -m pytest -m integration tests/integration
```

PowerShell:

```powershell
$env:AWS_AI_TOOLKIT_RUN_INTEGRATION = "1"
$env:AWS_REGION = "us-east-1"
$env:BEDROCK_INTEGRATION_MODEL_ID = "your-enabled-model-id"
$env:BEDROCK_INTEGRATION_GUARDRAIL_ID = "your-guardrail-id"
python -m pytest -m integration tests/integration
```

To verify collection without invoking AWS, run `python -m pytest --collect-only -q tests/integration` with the opt-in variable unset.
