#!/usr/bin/env bash
# Render the identifier-only environment file consumed by the MLflow service.
set -euo pipefail
source "$(dirname "$0")/../fleet/env.sh"
require_vars AWS_REGION BUCKET AURORA_ENDPOINT AURORA_SECRET_ARN \
  AURORA_DB_USER AURORA_DB_NAME

out="${1:-build/aws/fl-rmc-mlflow.env}"
mkdir -p "$(dirname "$out")"
{
  printf 'AWS_REGION=%q\n' "$AWS_REGION"
  printf 'PRAXIS_ARTIFACT_BUCKET=%q\n' "$BUCKET"
  printf 'AURORA_ENDPOINT=%q\n' "$AURORA_ENDPOINT"
  printf 'AURORA_SECRET_ARN=%q\n' "$AURORA_SECRET_ARN"
  printf 'AURORA_DB_USER=%q\n' "$AURORA_DB_USER"
  printf 'AURORA_DB_NAME=%q\n' "$AURORA_DB_NAME"
} > "$out"
chmod 0600 "$out"
echo "Rendered $out (resource identifiers only; no database password)."

