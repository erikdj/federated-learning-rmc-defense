# Portable AWS experiment fleet

These files reproduce the AWS Batch, S3, MLflow, and self-healing execution
design without embedding the original research account's resource identifiers.
Start with [the setup guide](../../docs/harness/aws-setup.md). The architecture
and recovery contracts are documented in [architecture.md](../../docs/harness/architecture.md)
and [recovery.md](../../docs/harness/recovery.md).

All scripts load `.env.aws` through `fleet/env.sh`. AWS credentials are never
stored in that file; the AWS CLI and SDK use their normal credential chain, and
Batch/Lambda/EC2 use IAM roles. Generated deployment records land under
`build/aws/` and are excluded from version control.

Running a build script pushes an image. Running a deploy or provision script
creates or changes AWS resources and may incur charges. None of these operations
is part of the offline reproduction path.

