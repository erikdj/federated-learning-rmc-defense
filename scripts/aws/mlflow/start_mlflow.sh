#!/usr/bin/env bash
# Fetch the managed Aurora password at service start, then exec MLflow.
set -euo pipefail
: "${AWS_REGION:?set AWS_REGION in /etc/fl-rmc-mlflow.env}"
: "${AURORA_SECRET_ARN:?set AURORA_SECRET_ARN in /etc/fl-rmc-mlflow.env}"
: "${AURORA_ENDPOINT:?set AURORA_ENDPOINT in /etc/fl-rmc-mlflow.env}"
: "${AURORA_DB_USER:?set AURORA_DB_USER in /etc/fl-rmc-mlflow.env}"
: "${AURORA_DB_NAME:?set AURORA_DB_NAME in /etc/fl-rmc-mlflow.env}"
: "${PRAXIS_ARTIFACT_BUCKET:?set PRAXIS_ARTIFACT_BUCKET in /etc/fl-rmc-mlflow.env}"

export AWS_DEFAULT_REGION="$AWS_REGION"
export PGPASSWORD
PGPASSWORD="$(aws secretsmanager get-secret-value --secret-id "$AURORA_SECRET_ARN" \
  --query SecretString --output text | jq -er .password)"

exec /opt/mlflow/bin/mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri "postgresql://$AURORA_DB_USER@$AURORA_ENDPOINT:5432/$AURORA_DB_NAME" \
  --default-artifact-root "s3://$PRAXIS_ARTIFACT_BUCKET/mlflow/artifacts/" \
  --serve-artifacts

