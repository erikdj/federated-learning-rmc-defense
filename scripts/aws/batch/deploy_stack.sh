#!/usr/bin/env bash
# Deploy the Batch Spot/On-Demand stack. This does not submit an experiment.
set -euo pipefail
source "$(dirname "$0")/../fleet/env.sh"
require_vars AWS_REGION VPC_ID BATCH_SUBNETS MLFLOW_SG_ID BUCKET STACK_PREFIX \
  PRAXIS_PROJECT_TAG PRAXIS_OWNER_TAG PRAXIS_PURPOSE_TAG \
  SPOT_MAX_VCPUS ON_DEMAND_MAX_VCPUS JOB_VCPUS \
  JOB_MEMORY_MIB RAY_CPUS JOB_TIMEOUT_SECONDS INSTANCE_TYPES

image_file="${PRAXIS_TRAINING_IMAGE_FILE:-build/aws/training-image.json}"
[[ -f "$image_file" ]] || { echo "ERROR: missing $image_file; run ecr/build_push.sh" >&2; exit 1; }
image="$(jq -er '.uri + "@" + .digest' "$image_file")"
route_tables="$(aws_cli ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'RouteTables[].RouteTableId' --output text | tr '\t' ',')"
[[ -n "$route_tables" ]] || { echo "ERROR: no route tables found for $VPC_ID" >&2; exit 1; }

endpoint="$(aws_cli ec2 describe-vpc-endpoints \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=service-name,Values=com.amazonaws.$AWS_REGION.s3" \
  --query 'VpcEndpoints[0].VpcEndpointId' --output text)"
create_endpoint=true
[[ -n "$endpoint" && "$endpoint" != None ]] && create_endpoint=false

stack_name="$STACK_PREFIX-batch"
aws_cli cloudformation deploy --stack-name "$stack_name" \
  --template-file scripts/aws/batch/batch-stack.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    "ImageDigest=$image" "Subnets=$BATCH_SUBNETS" "VpcId=$VPC_ID" \
    "RouteTableIds=$route_tables" "BucketArn=arn:aws:s3:::$BUCKET" \
    "MlflowSgId=$MLFLOW_SG_ID" "CreateS3Endpoint=$create_endpoint" \
    "ProjectTag=$PRAXIS_PROJECT_TAG" "OwnerTag=$PRAXIS_OWNER_TAG" \
    "PurposeTag=$PRAXIS_PURPOSE_TAG" \
    "SpotMaxVcpus=$SPOT_MAX_VCPUS" "OnDemandMaxVcpus=$ON_DEMAND_MAX_VCPUS" \
    "JobVcpus=$JOB_VCPUS" "JobMemoryMiB=$JOB_MEMORY_MIB" "RayCpus=$RAY_CPUS" \
    "JobTimeoutSeconds=$JOB_TIMEOUT_SECONDS" "InstanceTypes=$INSTANCE_TYPES"

mkdir -p build/aws
aws_cli cloudformation describe-stacks --stack-name "$stack_name" \
  --query 'Stacks[0].Outputs' --output json | tee build/aws/batch-stack-outputs.json
