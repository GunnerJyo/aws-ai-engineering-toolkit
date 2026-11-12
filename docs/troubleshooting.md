# Troubleshooting

- **AccessDeniedException:** verify the execution identity's scoped `bedrock:InvokeModel` or `bedrock:ApplyGuardrail` permission.
- **Model access or validation errors:** check the model ID/inference profile, enabled model access, supported region, and request parameter constraints.
- **Credential errors:** use a supported boto3 credential provider; never put keys in `.env` or source control.
- **Wrong vector dimensions:** keep the Titan embedding dimension consistent for every indexed and query vector.
- **Unexpected retrieval:** inspect chunk boundaries and metadata; the included store uses cosine similarity only and has no persistence or filtering.
- **Charges:** local unit tests are offline; running AWS examples makes service requests.
