#!/usr/bin/env bash
# Deploy MLflow finalizer/reaper resources. This does not submit an experiment.
set -euo pipefail
source "$(dirname "$0")/../fleet/env.sh"
require_vars AWS_REGION VPC_ID BATCH_SUBNETS MLFLOW_SG_ID BUCKET STACK_PREFIX \
  MLFLOW_TRACKING_URI PRAXIS_PROJECT_TAG PRAXIS_OWNER_TAG PRAXIS_PURPOSE_TAG

image_file="${PRAXIS_SELFHEAL_IMAGE_FILE:-build/aws/selfheal-image.json}"
batch_file="${PRAXIS_BATCH_OUTPUT_FILE:-build/aws/batch-stack-outputs.json}"
[[ -f "$image_file" ]] || { echo "ERROR: missing $image_file" >&2; exit 1; }
[[ -f "$batch_file" ]] || { echo "ERROR: missing $batch_file" >&2; exit 1; }
image="$(jq -er '.uri + "@" + .digest' "$image_file")"
job_queue_arn="$(jq -er '.[] | select(.OutputKey == "JobQueueArn") | .OutputValue' "$batch_file")"

endpoint="$(aws_cli ec2 describe-vpc-endpoints \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=service-name,Values=com.amazonaws.$AWS_REGION.batch" \
  --query 'VpcEndpoints[0].VpcEndpointId' --output text)"
create_batch_endpoint=true
[[ -n "$endpoint" && "$endpoint" != None ]] && create_batch_endpoint=false

stack_name="$STACK_PREFIX-selfheal"
aws_cli cloudformation deploy --stack-name "$stack_name" \
  --template-file scripts/aws/selfheal/selfheal-stack.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides "LambdaImageDigest=$image" "JobQueueArn=$job_queue_arn" \
    "Subnets=$BATCH_SUBNETS" "VpcId=$VPC_ID" "MlflowSgId=$MLFLOW_SG_ID" \
    "BucketName=$BUCKET" "BucketArn=arn:aws:s3:::$BUCKET" \
    "MlflowUri=$MLFLOW_TRACKING_URI" "CreateBatchEndpoint=$create_batch_endpoint" \
    "ProjectTag=$PRAXIS_PROJECT_TAG" "OwnerTag=$PRAXIS_OWNER_TAG" \
    "PurposeTag=$PRAXIS_PURPOSE_TAG"

mkdir -p build/aws
aws_cli cloudformation describe-stacks --stack-name "$stack_name" \
  --query 'Stacks[0].Outputs' --output json | tee build/aws/selfheal-stack-outputs.json
