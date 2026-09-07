# Evidence provenance and limits

The evidence package records the source revision and content hashes for each
artifact in [`provenance.json`](../../reproduction/evidence/provenance.json).
The current H1 read uses development-frozen risk cuts and the H2′ read includes
the window-aware leave-one-attack-out sensitivity. No manuscript, working-paper,
private planning report, cloud identifier, or historical infrastructure
configuration is part of the package.

H1 includes the current held-out read and its development-frozen thresholds.
H2, the H2 operating cuts, and H4 are byte-for-byte copies of their final JSON
artifacts. H2′ retains its primary P1/P2 values and verdict and adds the reported
window-aware sensitivity. The public H2′ copy replaces its historical bucket,
custody-record path, and temporary assembly-map path with
explicit non-distributed markers and points its protocol reference at the public
study guide. H3 retains every result and run identity but
replaces ten local temporary paths with `results/<basename>`. The original and
public hashes are recorded in
[`provenance.json`](../../reproduction/evidence/provenance.json).

The public H2′ helper modules retain the frozen feature builder, blend
constants, AUC primitive, and related numerical functions exactly, while
omitting each source script's unused standalone report entry point. The module
documentation is public-specific, and the unused historical output path was
removed from the feature-builder module. Original source hashes, public hashes,
and the extraction details are recorded in
[`provenance.json`](../../reproduction/evidence/provenance.json). Golden input
and expected-output bytes remain unchanged. The scorer imports these files as
protocol dependencies.

The following raw inputs are not distributed in this release:

- the 25 H1 detector-training signal logs from EXP-041 and EXP-046;
- 200 EXP-048 unit results and signal logs;
- the 50-cell EXP-051/053 signal corpus and its historical custody map;
- EXP-060 extracted events and per-unit result inputs;
- the 540 EXP-064/065 per-unit result inputs.

The absence has different consequences by artifact. The final H1, H2, H2′ and
H3 JSONs contain the statistics needed to check their frozen acceptance rules.
The H4 JSON additionally includes all 40 primary paired reductions, so its four
medians and exact Wilcoxon tests can be recomputed directly. None of the final
JSONs should be represented as raw experimental output. Re-executing the source
scorers from raw logs requires fresh runs or a future release of the historical
custody corpora.

The v2 fingerprint calibration file has public-only metadata and unchanged
numeric matrices. Its public SHA-256 is pinned by
`CALIBRATION_ARTIFACT_SHA256` in
[`flowerfl/fingerprint_registry.py`](../../flowerfl/fingerprint_registry.py).
The threshold constants in that file are 18.639429816855873 for validation and
26.466874982783164 for adjudicating. The H2′ serving bundle is pinned by the
SHA-256 of [`manifest_v2.json`](../../data/h4_serving/manifest_v2.json),
`3bfeefb45700dff2e02a14acb0de4acfadcc717a7b82d6c246d44a8f44fa062c`.

The H3 threshold and covariance are fitted on even-numbered partitions for the
adjudicating cohort. Feature screening reads all 20 partitions, and the pre-lock
homogeneity check also reads the odd partitions used for scoring. Fresh test
seeds therefore test the fixed instrument on new runs of known partitions;
they do not establish generalization to devices unseen during feature screening.

The historical MLflow service is private. This release provides neither a public
tracking server nor the complete historical raw run corpus. Its verification
commands read the bundled evidence and instrument artifacts; they do not replay
every historical run or establish remote backend completeness.

The frozen H2′ golden gate did not reproduce its expected feature hash on the
release-preparation host across the available OpenBLAS core settings. The gate
remains mandatory and unchanged. The bundled evidence contains the primary
results and the reported window-aware sensitivity; producing a new raw H2′
score requires a host where the unmodified golden test passes under the pinned
NumPy and scikit-learn versions. Inspecting the evidence or passing the software
suite alone does not establish that compatibility.

Archived `source_path` and protocol-authority strings in provenance and frozen
artifacts identify the original records; they are not links to distributed
files. Frozen models, seed files, and protocol fixtures retain those strings
where changing them would change their pinned bytes. The public
[study guide](experiments.md) and [protocol helpers](../../reproduction/protocol/)
provide the methods and executable definitions needed by the shipped scorers.
