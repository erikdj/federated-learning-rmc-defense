# Public release 0.1.1

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

H1 uses development-frozen upper-tail risk thresholds, with primary and
secondary results recorded separately. H2′ includes the window-aware
leave-one-attack-out sensitivity alongside unchanged primary P1/P2 results.
The study guides document the implemented calibration, scenario schedules,
aggregation rules and evaluation limits.

Release changes include packaging all three runtime packages, explicit data/resource
configuration, AWS region resolution, committed-source image builds, removing
import-time log-directory creation, and making software tests independent of
private infrastructure. Matrix launches detect the current Git branch, accept
an explicit branch and support `--no-push`. Experiment scaffolding includes a
complete public template. The default software suite excludes the separate
mandatory H2′ golden protocol gate.

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

## Dataset notices

The processed dataset's [DataCite DOI metadata](https://api.datacite.org/dois/10.21227/mbc1-1h68)
names CC BY 4.0. The [IEEE DataPort page](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)
also grants academic research use indefinitely and asks commercial users to
obtain permission from the lead author. Both notices are recorded in
[DATASET_LICENSE.md](../data/DATASET_LICENSE.md). This release does not resolve
their differing commercial-use conditions or claim separate permission.

The v2 data archive embeds the updated notice. All 23 research-data members
(Parquet files, feature manifest and metadata) are byte-identical to v1.
The software remains MIT licensed.

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

H1's criteria remain unmet at the development-frozen cuts. H3's simulated
identity construct, all-device screening and small honest sample, and unresolved
task-row overlap for utility interpretation are described in the study guides.
H4′ remains proposed and unrun.
