#!/usr/bin/env bash
# Build the training image, push it, and record the immutable registry digest.
set -euo pipefail
source "$(dirname "$0")/../fleet/env.sh"
require_vars AWS_REGION ACCOUNT_ID ECR_REPO

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
commit="$(git rev-parse HEAD)"
short_commit="$(git rev-parse --short HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked files are modified; commit them before building a custody image." >&2
  exit 1
fi

registry="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
uri="$registry/$ECR_REPO"
aws_cli ecr describe-repositories --repository-names "$ECR_REPO" >/dev/null 2>&1 || \
  aws_cli ecr create-repository --repository-name "$ECR_REPO" \
    --image-scanning-configuration scanOnPush=true >/dev/null
aws_cli ecr get-login-password | docker login --username AWS --password-stdin "$registry"
# Build from the committed tree represented by the recorded commit. Streaming a
# git archive prevents untracked files (including local configuration) from
# entering the image while keeping the build context identical to the custody
# commit stamped below.
git archive --format=tar "$commit" | docker build --platform linux/amd64 \
  -f docker/Dockerfile --build-arg PRAXIS_RUNNER_COMMIT="$commit" \
  -t "$uri:$short_commit" -
docker push "$uri:$short_commit"
digest="$(aws_cli ecr describe-images --repository-name "$ECR_REPO" \
  --image-ids "imageTag=$short_commit" --query 'imageDetails[0].imageDigest' --output text)"

out="build/aws/training-image.json"
mkdir -p "$(dirname "$out")"
printf '{"uri":"%s","tag":"%s","digest":"%s","runner_commit":"%s"}\n' \
  "$uri" "$short_commit" "$digest" "$commit" | tee "$out"
echo "Immutable image: $uri@$digest"
