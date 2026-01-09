# Load environment variables from .env file
set -a
source .env
set +a

# Construct the backend and artifact URIs
export MLFLOW_BACKEND_URI="postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"
export MLFLOW_ARTIFACT_URI="s3://${S3_BUCKET_NAME}"
export MLFLOW_REGISTRY_URI=$MLFLOW_BACKEND_URI

# Set AWS credentials for accessing S3
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# Start MLflow server
mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --registry-store-uri "$MLFLOW_REGISTRY_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_URI" \
  --host 0.0.0.0 \
  --port 5000 \
  --no-serve-artifacts