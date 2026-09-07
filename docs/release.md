# Public release 0.1.0

This is a new source repository assembled from the final research implementation
and selected evidence. It has a new Git history. The private working repository,
manuscript, dissertation figures, advisor materials, assistant memory, planning
files, and intermediate outputs are not distributed.

## Included

- Flower/PyTorch research runtime, detectors, fingerprinting and scenario inputs.
- Experiment CLI, S3 commit protocol, MLflow enrichment, Batch refills and tracking repair.
- Portable AWS configuration, deployment templates and container build scripts.
- Selected final numerical records, frozen models, seeds, splits and scoring tools.
- New architecture, recovery, data preparation and reproduction documentation.
- A separately licensed, checksum-verified processed dataset release download.

## Changes for public use

Scientific numeric parameters and scoring constructions are retained. Release
changes include packaging all three runtime packages, explicit data/resource
configuration, AWS region resolution, committed-source image builds, removing
import-time log-directory creation, and making software tests independent of
private infrastructure.

Obsolete standalone PoCs, historical tuning/fork analyses, one-off probes,
baseline-copy wrappers and unused fitted variants are excluded. Operational
progress, warnings, signal logs and provenance remain because recovery and
analysis consume them.

[data/public-artifact-provenance.json](../data/public-artifact-provenance.json)
records metadata redactions and both original and public checksums. Numerical
leaves in the three affected artifacts were compared with the original committed
files and are unchanged. The H3 registry binds the public calibration checksum.
The active H4 v2 serving model, feature list, cuts and manifest retain their
original bytes. Final evidence provenance is documented
[separately](reproduction/provenance.md).

## Verification and limits

Run the payload and evidence checks from the repository root:

```bash
python scripts/release/verify.py
python reproduction/verify_published_results.py
sha256sum -c reproduction/evidence/SHA256SUMS
sha256sum -c reproduction/protocol/SHA256SUMS
```

The release manifest binds every tracked payload file except itself and rejects
unlisted tracked files. It detects accidental changes; authenticity still depends
on obtaining the manifest and source from the intended repository/release.

The offline software suite, shell syntax checks, CloudFormation validation,
package build and CLI smoke checks were exercised during release preparation.
The exact processed-data archive was verified against every file checksum,
Parquet schema, row count and frozen holdout fingerprint. New cloud infrastructure
and full experiment fleets were not deployed or rerun during publication.

The strict H2′ golden gate remains a known host-compatibility limit: canonical
floating-point bytes differ on the preparation machine. The gate was preserved,
not converted to a tolerance or skipped by the adjudicator. Run the separate
[mandatory protocol check](installation.md#strict-h2-numerical-gate) before
scoring a new H2′ corpus. Software test success is not evidence that this gate
passes on a particular host.

Original raw per-unit experiment corpora and a public MLflow server are not
included. The bundled numerical reads support result inspection and consistency
checks; full raw-to-verdict reproduction requires new fleets or the corresponding
raw artifacts. The exact dataset, scientific matrices, runtime, calibrated
instruments and scorer entry points are supplied for that purpose.

H1's operating-point mismatch, H3's simulated identity construct and small honest
sample, and unresolved task-row overlap for utility interpretation are described
in the study guides. H4′ remains proposed and unrun. This release does not revise
any recorded hypothesis verdict.
