# Daily Trading Workflow on AWS Lambda + EventBridge (Container Image)

This document describes how to deploy `agentcore.lambda_handler` as an AWS Lambda
function using a container image, and trigger it daily via Amazon EventBridge.

## 1. Prerequisites

- AWS account ID: **291573578422**
- AWS region: **eu-central-1** (Frankfurt)
- ECR repository: `multi-trading-aws-agent-orquestrator` (you can create it if it
  does not already exist)
- Docker installed and logged in to AWS ECR
- IAM permissions to create ECR repos, Lambda functions, and EventBridge rules

## 2. Build and push the Lambda container image

From the root of this repository (where `agentcore.py` and `Dockerfile` live):

```bash
REGION=eu-central-1
ACCOUNT_ID=291573578422
REPO=multi-trading-aws-agent-orquestrator

# 1) Create ECR repo (one-time)
aws ecr create-repository --repository-name $REPO --region $REGION || true

# 2) Authenticate Docker to ECR
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin \
    $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 3) Build, tag, and push the image
docker build -t $REPO .
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
```

Resulting image URI:

```text
291573578422.dkr.ecr.eu-central-1.amazonaws.com/multi-trading-aws-agent-orquestrator:latest
```

## 3. Lambda function configuration

Create a new Lambda function using **container image** packaging.

- **Image URI:**
  `291573578422.dkr.ecr.eu-central-1.amazonaws.com/multi-trading-aws-agent-orquestrator:latest`
- **Handler:** (for container images, the handler is defined in the image CMD)
  - The Dockerfile already sets: `CMD ["agentcore.lambda_handler"]`
- **Runtime:** `python3.12` (implicit for this base image)
- **Timeout:** e.g. `900` seconds
- **Memory:** e.g. `1024` MB

### Environment variables

Set at least:

- `OPENAI_API_KEY` – your OpenAI API key (ideally from Secrets Manager/SSM)
- `SHARE_LIST` – optional default list of tickers if the EventBridge event
  does not pass one, e.g.: `AAPL,GOOGL,MSFT`

The Lambda handler is implemented in `agentcore.py` as:

```python
def lambda_handler(event, context):
    ...
```

It expects an event of the form:

```json
{
  "shares": ["AAPL", "GOOGL", "MSFT"]
}
```

If `shares` is not provided, it falls back to `SHARE_LIST`.

## 4. EventBridge schedule rule

Create an EventBridge rule to trigger the Lambda daily (example: weekdays at
20:00 UTC):

- **Schedule expression:**

  ```text
  cron(0 20 ? * MON-FRI *)
  ```

- **Target:** your Lambda function created above.
- **Input:** constant JSON, for example:

  ```json
  {
    "shares": ["AAPL", "GOOGL", "MSFT"]
  }
  ```

EventBridge will pass this payload as the `event` argument to `lambda_handler`.

## 5. Optional: Terraform example

If you use Terraform, you can model the Lambda and EventBridge rule roughly as
follows (adjust names as needed):

```hcl
provider "aws" {
  region = "eu-central-1"
}

resource "aws_iam_role" "trading_lambda_role" {
  name = "trading-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action   = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "trading_lambda_logs" {
  role       = aws_iam_role.trading_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "trading_lambda" {
  function_name = "daily-trading-workflow"
  role          = aws_iam_role.trading_lambda_role.arn
  package_type  = "Image"
  image_uri     = "291573578422.dkr.ecr.eu-central-1.amazonaws.com/multi-trading-aws-agent-orquestrator:latest"

  timeout     = 900
  memory_size = 1024

  environment {
    variables = {
      OPENAI_API_KEY = "REPLACE_WITH_SECRET_OR_USE_SSM"
      SHARE_LIST     = "AAPL,GOOGL,MSFT"
    }
  }
}

resource "aws_cloudwatch_event_rule" "daily_trading" {
  name                = "daily-trading-schedule"
  schedule_expression = "cron(0 20 ? * MON-FRI *)"
}

resource "aws_cloudwatch_event_target" "daily_trading_target" {
  rule      = aws_cloudwatch_event_rule.daily_trading.name
  target_id = "daily-trading-lambda"
  arn       = aws_lambda_function.trading_lambda.arn

  input = jsonencode({
    shares = ["AAPL", "GOOGL", "MSFT"]
  })
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trading_lambda.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_trading.arn
}
```

This Terraform snippet is illustrative; adapt resource names, tags, and
security settings to your environment.
