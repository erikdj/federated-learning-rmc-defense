#!/usr/bin/env bash
# Build and push the slim MLflow reconciliation Lambda image.
set -euo pipefail
source "$(dirname "$0")/../fleet/env.sh"
require_vars AWS_REGION ACCOUNT_ID ECR_REPO

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
commit="$(git rev-parse HEAD)"
short_commit="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked files are modified; commit before building a custody image." >&2
  exit 1
fi

metadata="${PRAXIS_DATASET_METADATA:-data/edge_full_20/metadata.json}"
[[ -f "$metadata" ]] || {
  echo "ERROR: missing $metadata; acquire the public dataset metadata before building." >&2
  exit 1
}

context_dir="$(mktemp -d)"
trap 'rm -rf "$context_dir"' EXIT
git archive HEAD praxis_exp docker/entrypoint.py docker/Dockerfile.lambda | tar -x -C "$context_dir"
install -D "$metadata" "$context_dir/data/edge_full_20/metadata.json"

repository="$ECR_REPO-selfheal"
registry="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
uri="$registry/$repository"
aws_cli ecr describe-repositories --repository-names "$repository" >/dev/null 2>&1 || \
  aws_cli ecr create-repository --repository-name "$repository" \
    --image-scanning-configuration scanOnPush=true >/dev/null
aws_cli ecr get-login-password | docker login --username AWS --password-stdin "$registry"
docker build --platform linux/amd64 -f "$context_dir/docker/Dockerfile.lambda" \
  -t "$uri:$short_commit" "$context_dir"
docker push "$uri:$short_commit"
digest="$(aws_cli ecr describe-images --repository-name "$repository" \
  --image-ids "imageTag=$short_commit" --query 'imageDetails[0].imageDigest' --output text)"
metadata_sha256="$(sha256sum "$metadata" | awk '{print $1}')"

mkdir -p build/aws
printf '{"uri":"%s","tag":"%s","digest":"%s","runner_commit":"%s","metadata_sha256":"%s"}\n' \
  "$uri" "$short_commit" "$digest" "$commit" "$metadata_sha256" | tee build/aws/selfheal-image.json

