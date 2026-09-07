# Reproducing the reported experiments

This directory separates two reproducibility levels. The files in
[`reproduction/evidence`](../../reproduction/evidence) are the final
machine-readable reads used for the reported claims. They let a reviewer inspect
the per-seed or per-component values and re-evaluate each acceptance rule with
[`verify_published_results.py`](../../reproduction/verify_published_results.py).
The historical per-run result and signal corpora are not distributed, so a
raw-to-verdict rerun starts by producing a fresh fleet from the supplied matrix
descriptors.

| Claim | Historical experiment set | Result | Public evidence |
|---|---|---|---|
| H1 | EXP-041, EXP-046, EXP-048 | Not met | `h1-heldout.json` |
| H2 | EXP-048 | Falsified | `h2-heldout.json`, frozen cuts |
| H2′ | EXP-051 plus two EXP-053 refill cells | Confirmed | `h2prime-confirmatory.json` |
| H3 | EXP-059 validation, EXP-060 adjudication | Pass | `h3-confirmatory.json` |
| H4 | EXP-064 plus the EXP-065 arm-4 companion | Falsified | `h4-confirmatory.json` |
| H4′ | none | Draft, unratified, unrun | no result artifact |

Install the pinned CPU environment by following the
[installation guide](../installation.md), then run the evidence-level checks:

```bash
conda run -n flowerfl python reproduction/verify_published_results.py
sha256sum -c reproduction/evidence/SHA256SUMS
sha256sum -c reproduction/protocol/SHA256SUMS
```

The fresh-run path is:

1. acquire and prepare Edge-IIoTset as described in [Data](data.md);
2. select the matrix JSON in [`reproduction/configs`](../../reproduction/configs);
3. configure the execution environment with the
   [AWS setup guide](../harness/aws-setup.md), then render a new experiment
   document with the queue, job definition, and immutable image digest;
4. launch and retain every result, signal, and completion marker under the new
   experiment identity;
5. stage the completed local corpus and run the scorer commands in
   [Experiments and analysis](experiments.md).

The matrix descriptors contain scientific inputs only. They contain no
historical account identifiers, storage locations, or job resources. The
renderer refuses the H4′ descriptor because that protocol has not been
ratified and no fresh seed set exists.

[`provenance.json`](../../reproduction/evidence/provenance.json) records the
source commit, original repository-relative paths and hashes, public hashes,
and every public redaction. [Provenance and limits](provenance.md) explains
what can and cannot be reconstructed from this release.
