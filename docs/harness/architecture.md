# Experiment harness architecture

The harness maps a reviewed experiment matrix to independent AWS Batch units.
S3 is the durable system of record for completion; MLflow is the searchable
metadata and visualization layer. Training images and self-heal images are
addressed by registry digest.

```mermaid
flowchart LR
    O[Operator: praxis CLI] -->|writes immutable manifest| S3[(S3 sweep namespace)]
    O -->|creates parent run| M[(MLflow on EC2 + Aurora)]
    O -->|submits array with digest| B[AWS Batch queue]
    B --> SPOT[Spot compute environment]
    B --> OD[On-Demand fallback]
    SPOT --> C[One container per matrix unit]
    OD --> C
    C -->|sync public dataset| S3
    C -->|run Flower simulation| R[Scientific runner]
    R -->|result JSON + signal JSONL| C
    C -->|payloads, verify sizes, marker last| S3
    C -->|params, metrics, status, artifacts| M
    E[EventBridge terminal event] --> F[Finalizer Lambda]
    T[20-minute schedule] --> P[Reaper Lambda]
    F -->|read S3; repair MLflow| M
    P -->|read S3 and Batch; repair MLflow| M
```

Each array index resolves through the canonical S3 manifest to a deterministic
unit ID derived from defense, scenario, execution mode, seed, and optional
replicate. The container receives the experiment ID, artifact bucket, MLflow
URI, image digest, launch commit, scenario directory, and parent run ID as Batch
environment values. It does not receive static AWS credentials.

The deployment has four trust boundaries:

- The operator identity may build images, write manifests, create MLflow parent
  runs, and submit arrays.
- The Batch job role can read the dataset and read/write experiment artifacts in
  the configured bucket.
- The MLflow host role can fetch the managed Aurora secret and read/write its S3
  artifact prefix.
- The self-heal Lambda role can read sweep state, repair MLflow artifacts, and
  query Batch status. It cannot submit jobs.

The CloudFormation templates intentionally accept VPC, subnet, security-group,
bucket, capacity, ownership, and image values as parameters. No original AWS
account, address, profile, hostname, or personal tag is present in the release.

