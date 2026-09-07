# Evidence provenance and limits

The curated evidence was read from source commit
`a8605f60b1d75c0b76f42a4aea4023533af3f54d`. No manuscript, working-paper,
private planning report, cloud identifier, or historical infrastructure
configuration is part of the evidence package.

H1, H2, the H2 operating cuts, and H4 are byte-for-byte copies of the final JSON
artifacts. H2′ retains every scientific and numerical field but replaces its
historical bucket, custody-record path, and temporary assembly-map path with
explicit non-distributed markers. H3 retains every result and run identity but
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

The following inputs are absent from both the source Git checkout and this
release:

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

The local research tracking service was unavailable during release curation.
The audit therefore could not independently enumerate historical backend runs
or check for unregistered artifacts. Git-tracked final evidence, frozen model
bytes, scorer inputs, experiment matrices, and committed custody statements
were inspected; remote backend completeness remains outside this release's
verifiable boundary.

The frozen H2′ golden gate did not reproduce its expected feature hash on the
release-curation host across the available OpenBLAS core settings. Because that
gate is mandatory, no H2′ live scoring was performed during curation and the
gate was not weakened. The public evidence is the curated historical terminal
artifact; a new raw H2′ score is valid only on a host where the unmodified
golden test passes under the pinned NumPy and scikit-learn versions.
