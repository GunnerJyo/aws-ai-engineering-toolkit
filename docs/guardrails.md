# Bedrock Guardrails

`apply_guardrail(client, guardrail_id, version, text, source="INPUT")` sends `ApplyGuardrail` independently of a foundation model request. Use `source="INPUT"` for user text and `source="OUTPUT"` for generated text. Inspect `intervened`, `outputs`, `action_reason`, and `assessments`; never assume that evaluation passed without checking `action`. Grant `bedrock:ApplyGuardrail` on the intended guardrail. Follow the [AWS independent ApplyGuardrail guide](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html). Charges may apply.
