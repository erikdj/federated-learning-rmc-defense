# AWS setup

This is an optional distributed execution path. Local scientific reproduction
does not require AWS. The commands below create billable resources when you run
them; this release does not run them automatically.

## Prerequisites

Install AWS CLI v2, Docker with BuildKit, Git, `jq`, and a shell with Bash 4 or
newer. Configure an AWS identity with permission to manage ECR, Batch,
CloudFormation, EC2, IAM, RDS, Lambda, EventBridge, Secrets Manager, and the
chosen S3 bucket.

Prepare a VPC with subnets in at least two Availability Zones. Batch instances
need routes or VPC endpoints for ECS, ECR, CloudWatch Logs, and S3. The templates
can create an S3 gateway endpoint and a Batch interface endpoint, but they do
not create the VPC or its general egress path. Create one shared workload
security group for the MLflow host, Batch instances, and self-heal Lambda. Add a
self-referencing TCP 5000 ingress rule so only members of that group can reach
MLflow. Do not expose the unauthenticated MLflow server to the public Internet.

## Configure identifiers

```bash
cp scripts/aws/config.example.env .env.aws
$EDITOR .env.aws
export PRAXIS_AWS_CONFIG=.env.aws
```

`.env.aws` is ignored by Git. It contains resource identifiers and capacity
choices, never access keys or database passwords. The database password stays
in the RDS-managed Secrets Manager secret.

Create and secure the artifact bucket before deployment. Enable versioning and
default encryption, then upload the public dataset in this layout:

```text
s3://BUCKET/data/edge_full_20/
  client_*.parquet
  metadata.json
  features.json
```

Dataset acquisition and integrity checks are described by the repository's data
guide. The training container synchronizes this prefix at startup.

## Provision Aurora

Fill `AURORA_CLUSTER_ID`, `AURORA_INSTANCE_ID`, `AURORA_DB_USER`, and
`AURORA_DB_NAME`, then run:

```bash
bash scripts/aws/aurora/provision.sh
```

The script creates a Serverless v2 PostgreSQL cluster with encrypted storage and
an RDS-managed password. It writes the endpoint and secret ARN to
`build/aws/aurora.json`. Copy those two identifiers into `.env.aws`. The example
uses a broadly supported 0.5 minimum ACU. Where the selected engine version and
region support auto-pause, set `AURORA_MIN_ACU=0`; the script then applies
`AURORA_AUTO_PAUSE_SECONDS`.

## Bootstrap MLflow

Create an Amazon Linux 2023 EC2 instance in `MLFLOW_SUBNET` with the MLflow
security group and a private address reachable from Batch. Its instance role
needs `secretsmanager:GetSecretValue` for the generated Aurora secret and S3
read/write access to `s3://BUCKET/mlflow/artifacts/`. SSM Session Manager access
is recommended instead of inbound SSH.

Render and install the identifier-only environment file, then run the bootstrap
from a checkout on the host:

```bash
bash scripts/aws/mlflow/render_host_env.sh
# transfer build/aws/fl-rmc-mlflow.env through SSM or your approved channel
sudo install -m 0600 build/aws/fl-rmc-mlflow.env /etc/fl-rmc-mlflow.env
sudo bash scripts/aws/mlflow/bootstrap.sh
```

The service fetches the database password from Secrets Manager at every start;
the password does not appear in the process arguments or tracked files. Record
the host's private URI as `MLFLOW_TRACKING_URI` in `.env.aws`. If you expose the
UI through private DNS or a proxy, add MLflow's allowed-host and CORS environment
variables to `/etc/fl-rmc-mlflow.env`.

## Build and deploy the fleet

Commit the exact source tree first. Both image builders refuse a dirty tracked
tree because the commit baked into the image is part of the custody record.

```bash
bash scripts/aws/ecr/build_push.sh
bash scripts/aws/batch/deploy_stack.sh
```

The Batch output file in `build/aws/` supplies the queue and job-definition ARNs
for experiment design documents. Start with small `SPOT_MAX_VCPUS` and
`ON_DEMAND_MAX_VCPUS` values for validation. The On-Demand environment is a
fallback and can create unexpected cost if Spot capacity is scarce.

## Connect the deployment to the CLI

Export the site values before rendering or launching an experiment. Sourcing
`.env.aws` sets `MLFLOW_TRACKING_URI` to the private address used by Batch and
self-heal. Save that value as the container URI **before** replacing
`MLFLOW_TRACKING_URI` with an address the operator machine can reach:

```bash
set -a
source .env.aws
set +a

export PRAXIS_CONTAINER_MLFLOW_URI="$MLFLOW_TRACKING_URI"
export MLFLOW_TRACKING_URI=http://localhost:5001  # for example, an SSM port forward
export AWS_PROFILE=replace-with-your-cli-profile
export AWS_REGION="$AWS_REGION"
export PRAXIS_ARTIFACT_BUCKET="$BUCKET"

export JOB_QUEUE="$(jq -er \
  '.[] | select(.OutputKey == "JobQueueArn") | .OutputValue' \
  build/aws/batch-stack-outputs.json)"
export JOB_DEFINITION="$(jq -er \
  '.[] | select(.OutputKey == "JobDefinitionArn") | .OutputValue' \
  build/aws/batch-stack-outputs.json)"
export IMAGE_DIGEST="$(jq -er '.digest' build/aws/training-image.json)"
```

`AWS_PROFILE` authenticates the operator-side CLI. `AWS_REGION` takes
precedence over `AWS_DEFAULT_REGION`; both the S3 and Batch clients use the
resolved region. `PRAXIS_ARTIFACT_BUCKET` is the CLI name for the harness's
`BUCKET` value. The three variables derived from `build/aws/` feed
`reproduction/render_experiment_doc.py`; `IMAGE_DIGEST` is the bare
`sha256:...` value expected by both the renderer and `praxis exp launch-matrix`.

After acquiring `data/edge_full_20/metadata.json`, build and deploy the
reconciler:

```bash
bash scripts/aws/selfheal/build_push_lambda.sh
bash scripts/aws/selfheal/deploy_stack.sh
```

The deployment creates a terminal-array EventBridge rule and a scheduled reaper.
It does not submit an experiment. Launches still go through `praxis exp
launch-matrix` so the manifest, Git tag, image digest, MLflow parent, and Batch
array are created as one reviewed lifecycle.

If the VPC already has a Batch interface endpoint, the deploy script reuses it.
Confirm that endpoint's security group admits HTTPS from the self-heal Lambda
security group; the template can enforce that rule only for an endpoint it
creates.

Before real compute, validate a single unit, confirm the S3 result/signal/marker
contract, inspect its MLflow record, and deliberately test a host-interruption
retry in a disposable experiment namespace.
