# Distributed experiment harness

The optional AWS harness runs reviewed experiment matrices as independent,
digest-pinned Batch units while keeping S3 as the durable completion record.

- [Architecture](architecture.md) explains the components, data flow, and trust
  boundaries.
- [AWS setup](aws-setup.md) describes portable configuration, Aurora and MLflow
  bootstrap, image builds, and CloudFormation deployment.
- [Recovery](recovery.md) states the implemented retry, commit, refill, and
  self-heal behavior, including the absence of mid-unit checkpointing.

The deployment files live under [`scripts/aws/`](../../scripts/aws/). They
contain no account-specific resource identifiers and never store AWS or database
credentials.

