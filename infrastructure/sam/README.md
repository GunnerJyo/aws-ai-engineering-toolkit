# AWS SAM Lambda example

The template deploys an unauthenticated API Gateway endpoint and Lambda function. It grants `bedrock:InvokeModel` only on the model ARN supplied at deploy time. Restrict or protect the endpoint before using it with real data.

Prerequisites: AWS SAM CLI, Docker only if using container builds, configured AWS credentials, model access, and an exact model or inference-profile ARN supported in the selected region. Deployment and inference may incur AWS charges.

```bash
sam validate --template-file infrastructure/sam/template.yaml
sam build --template-file infrastructure/sam/template.yaml
sam deploy --template-file .aws-sam/build/template.yaml --guided
```

Choose the deployment region with the SAM guided deployment prompt; Lambda supplies `AWS_REGION` automatically. Provide `BedrockModelArn` and `BedrockModelId`. Test `POST /prompt` with JSON `{"prompt":"Hello"}`. The example has no authentication, rate limiting, or WAF configuration. After review, remove the stack using its deployed stack name:

```bash
aws cloudformation delete-stack --stack-name YOUR_STACK_NAME
```
