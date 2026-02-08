FROM public.ecr.aws/lambda/python:3.12

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Lambda handler
CMD ["agentcore.lambda_handler"]
