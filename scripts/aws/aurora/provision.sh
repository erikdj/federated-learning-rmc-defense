#!/usr/bin/env bash
# Provision an Aurora Serverless v2 PostgreSQL backend for a new MLflow server.
set -euo pipefail
source "$(dirname "$0")/../fleet/env.sh"
require_vars AWS_REGION VPC_ID BATCH_SUBNETS MLFLOW_SG_ID STACK_PREFIX \
  AURORA_CLUSTER_ID AURORA_INSTANCE_ID AURORA_DB_USER AURORA_DB_NAME

engine_version="${AURORA_ENGINE_VERSION:-16.8}"
min_acu="${AURORA_MIN_ACU:-0.5}"
max_acu="${AURORA_MAX_ACU:-16}"
auto_pause_seconds="${AURORA_AUTO_PAUSE_SECONDS:-300}"
subnet_group="$STACK_PREFIX-aurora-subnets"
security_group_name="$STACK_PREFIX-aurora"
scaling="MinCapacity=$min_acu,MaxCapacity=$max_acu"
if [[ "$min_acu" == "0" || "$min_acu" == "0.0" ]]; then
  scaling+=",SecondsUntilAutoPause=$auto_pause_seconds"
fi

aurora_sg="$(aws_cli ec2 describe-security-groups \
  --filters "Name=group-name,Values=$security_group_name" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)"
if [[ -z "$aurora_sg" || "$aurora_sg" == None ]]; then
  aurora_sg="$(aws_cli ec2 create-security-group --group-name "$security_group_name" \
    --description 'PostgreSQL access from the MLflow security group' --vpc-id "$VPC_ID" \
    --query GroupId --output text)"
  aws_cli ec2 authorize-security-group-ingress --group-id "$aurora_sg" \
    --protocol tcp --port 5432 --source-group "$MLFLOW_SG_ID"
fi

IFS=',' read -r -a subnet_array <<< "$BATCH_SUBNETS"
if ! aws_cli rds describe-db-subnet-groups --db-subnet-group-name "$subnet_group" >/dev/null 2>&1; then
  aws_cli rds create-db-subnet-group --db-subnet-group-name "$subnet_group" \
    --db-subnet-group-description 'MLflow Aurora subnets' --subnet-ids "${subnet_array[@]}" >/dev/null
fi

if ! aws_cli rds describe-db-clusters --db-cluster-identifier "$AURORA_CLUSTER_ID" >/dev/null 2>&1; then
  aws_cli rds create-db-cluster --db-cluster-identifier "$AURORA_CLUSTER_ID" \
    --engine aurora-postgresql --engine-version "$engine_version" \
    --database-name "$AURORA_DB_NAME" --master-username "$AURORA_DB_USER" \
    --manage-master-user-password --storage-encrypted \
    --serverless-v2-scaling-configuration "$scaling" \
    --db-subnet-group-name "$subnet_group" --vpc-security-group-ids "$aurora_sg" >/dev/null
fi
if ! aws_cli rds describe-db-instances --db-instance-identifier "$AURORA_INSTANCE_ID" >/dev/null 2>&1; then
  aws_cli rds create-db-instance --db-instance-identifier "$AURORA_INSTANCE_ID" \
    --db-cluster-identifier "$AURORA_CLUSTER_ID" --engine aurora-postgresql \
    --db-instance-class db.serverless >/dev/null
fi

aws_cli rds wait db-instance-available --db-instance-identifier "$AURORA_INSTANCE_ID"
mkdir -p build/aws
aws_cli rds describe-db-clusters --db-cluster-identifier "$AURORA_CLUSTER_ID" \
  --query 'DBClusters[0].{endpoint:Endpoint,secret_arn:MasterUserSecret.SecretArn,engine_version:EngineVersion,cluster_arn:DBClusterArn}' \
  --output json | tee build/aws/aurora.json
echo "Copy endpoint and secret_arn from build/aws/aurora.json into .env.aws."
