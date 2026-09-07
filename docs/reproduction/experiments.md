# Experiments and analysis

All historical and fresh reproduction matrices use `persistent_optimizer`, a
2,000,000-sample per-client cap, 50 rounds, and one repeat. Scenario behavior
comes from the JSON files in [`rmc/scenarios`](../../rmc/scenarios); filenames
are identifiers and do not override their contents.

## Rendering and launching a fresh matrix

The JSON files in [`reproduction/configs`](../../reproduction/configs) contain
the exact scientific matrix and run extras without site-specific resources.
Complete the profile, artifact-store, tracking, queue, job-definition, and
container settings in the [AWS setup guide](../harness/aws-setup.md) first.
Render one into the experiment-document format after choosing a new experiment
ID and immutable execution resources:

```bash
conda run -n flowerfl python reproduction/render_experiment_doc.py \
  reproduction/configs/h1-h2-heldout.json \
  --exp-id EXP-101 \
  --job-queue "$JOB_QUEUE" \
  --job-definition "$JOB_DEFINITION" \
  --image-digest "$IMAGE_DIGEST" \
  --out docs/experiments/EXP-101-h1-h2-reproduction.md

git add docs/experiments/EXP-101-h1-h2-reproduction.md
git commit -m "Register EXP-101 reproduction matrix"

conda run -n flowerfl praxis exp launch-matrix EXP-101 \
  --image-digest "$IMAGE_DIGEST"
```

The renderer refuses existing output files, malformed experiment IDs and image
digests, non-launchable descriptors, matrix-size mismatches, unsupported
defenses, and documents rejected by the launch parser. Commit the document
before launch because the launcher enforces a clean, registered source state.
Choose a separate fresh experiment identity for each matrix. Preserve the
complete Cartesian product and retain one result JSON, one signal JSONL, and
one completion marker for every unit. A refill may supply only a missing cell;
record the cell-to-source decision before analysis so an incidental rerun cannot
silently replace a completed unit.

The H4 historical matrix needed one configuration companion. EXP-064 supplied
480 successful cells; EXP-065 supplied the 60 `Krum+TGE+FP` cells with explicit
`fp_cohort=validation` and `fp_registry_policy=flag_gated`. The combined
`h4-confirmatory.json` file is an assembly descriptor and the renderer refuses
it. Render `h4-confirmatory-main.json` and `h4-confirmatory-arm4.json` under two
fresh experiment IDs, commit both documents, launch both, and assemble exactly
one result per semantic cell.

## H1

The detector-training corpus is Krum+TGE × S0–S4 × development seeds
`[42, 137, 256, 314, 500]`, 25 cells, with train-only normalization. Historically
EXP-046 supplied S0–S3 and EXP-041 supplied S4. Refit the three k=3 GBDTs from a
directory containing that exact filename grid:

```bash
conda run -n flowerfl python scripts/h1_retrain_ramp3_dev_read.py \
  --signals-dir staging/h1-training/signals \
  --models-dir staging/h1-training/models \
  --out staging/h1-training/read.json \
  --read-label "fresh H1 reproduction fit" \
  --input-scope "Krum+TGE x S0-S4 x five development seeds; train-only normalization" \
  --trained-on "fresh reproduction corpus" \
  --provenance "local hash-pinned custody manifest"
```

The published held-out read uses the included frozen models and the Krum+TGE
subset of EXP-048: S0–S4 × ten seeds, 50 signal logs. Stage those filenames in
one directory and run:

```bash
conda run -n flowerfl python reproduction/analyze_h1.py \
  --signals-root staging/EXP-048/signals \
  --out staging/h1-recomputed.json \
  --check-against reproduction/evidence/h1-heldout.json
```

The three conjunctive criteria are C−S ≥ 0.05 in each of S2–S4, C−W ≥ 0.05 in
each of S2–S4, and one-sided paired exact Wilcoxon p ≤ 1/32 for both comparisons
in each scenario. H1 was not met: the C−S margins were 0.1074, 0.0289 and 0.0444;
the C−W margins were −0.1185, −0.0163 and 0.0911. The nominal 10% FPR threshold
was unstable on the small cold-start honest population; realized cell-level FPR
was about 0.88. Interpret the result as a frozen protocol result with that
operating-point limitation.

## H2

EXP-048 is four defenses (`Krum`, `TrustScore`, standalone `TGE`, `Krum+TGE`) ×
S0–S4 × seeds `[1009, 1733, 2521, 3299, 4127, 5051, 6079, 7177, 8231, 9337]`,
200 units, with train-only normalization. The all-rounds scorer uses frozen
10%-FPR cuts: Krum 0.18790065502712283, TrustScore 0.3677150011062622, TGE
0.3822, and Krum+TGE 0.0932.

The scorer expects the historical logical directory name `EXP-048`, regardless
of where the fresh fleet ran. Assemble a staging tree with `EXP-048/results`
and `EXP-048/signals`, containing exactly one file per cell, then run:

```bash
conda run -n flowerfl python scripts/analyze_h2_confirm_read.py \
  --data-root staging \
  --thresholds reproduction/evidence/h2-thresholds-leakfree-10pct.json \
  --out staging/h2-recomputed.json
```

H2 requires standalone TGE recall at least 0.85 in S4, a TGE advantage of at
least 0.05 over the best paired baseline in each of S2–S4, and one-sided paired
exact Wilcoxon p ≤ 1/32 against that best baseline in every scenario. It was
falsified. TGE S4 recall was 0.08937198067632851; its S2–S4 margins were
−0.24578, 0.20427 and −0.07536.

## H2′

The H2′ corpus is Krum+TGE × S0–S4 × the ten values in
[`h2prime_confirm_seeds.json`](../../data/h2prime_confirm_seeds.json), 50 signal
logs. The scorer sorts seeds ascending. For test seed i, the next cyclic seed is
the honest calibration seed and the following three cyclic seeds are fit seeds.
One GBDT is fit per held-out attack family after excluding that family. Each
scenario uses the linear 0.90 quantile of honest calibration scores and flags
only scores strictly greater than the cut. NumPy 2.2.6 and scikit-learn 1.7.2
are protocol pins.

Build a content-hashed assembly map. A single fresh source needs no sidecar:

```bash
conda run -n flowerfl python reproduction/build_h2prime_assembly_map.py \
  --source fresh=staging/h2prime/signals \
  --out staging/h2prime/assembly-map.json
```

For multiple sources, pass each as `--source label=directory` and provide a
JSON sidecar with a `cells` array whose entries contain `scenario`, `seed`, and
`source`. The builder refuses anything except the exact 5 × 10 cross-product
and hashes every selected file.

The frozen executor also requires the 50 exposed standalone-TGE logs from
EXP-048 for a mandatory reported secondary. Point the environment variable at
that complete S0–S4 × H2-seed directory, validate without reading the sealed
corpus, then score:

```bash
export H2PRIME_EXP048_SIG_DIR="$PWD/staging/EXP-048/signals-tge-only"

conda run -n flowerfl python scripts/adjudicate_h2prime.py \
  staging/h2prime/assembly-map.json --dry-run

conda run -n flowerfl python scripts/adjudicate_h2prime.py \
  staging/h2prime/assembly-map.json \
  --out staging/h2prime/adjudication.json
```

P1 requires mean blended leave-one-attack-out S4 recall ≥ 0.35 with the mean
realized blended FPR in `[0.08, 0.12]`. It passed at recall 0.502020202020202 and
FPR 0.1026039919490104. P2 requires the detector to strictly beat the per-seed
oracle maximum of Krum score, L2-to-median and cosine-to-median in at least nine
of ten seeds, with detector and selected-baseline pooled FPR both in the same
interval. It passed 10/10; exact sign p was 0.0009765625, detector FPR was
0.11551155115511551, and baseline FPR was 0.10843417675100843. H2′ is confirmed
because both components passed.

## H3

EXP-059 is the non-adjudicating corrected-instrument validation matrix: TGE and
TGE+FP × S3/S4 × the five public development seeds, 20 units, using
`identity_only`, validation cohort, and τ = 18.639429816855873. Its integrity
gates opened EXP-060. EXP-060 repeats the same arms and scenarios on seeds
`[11749, 19025, 66912, 75901, 93207]`, using adjudicating cohort,
`identity_only`, τ = 26.466874982783164, and odd-device holdout.

Extract events from the ten TGE+FP signal logs and pass the corresponding ten
TGE+FP result JSONs as provenance:

```bash
conda run -n flowerfl python scripts/extract_h3_events.py \
  --logs staging/EXP-060/signals/*__tge_fp__persistent_optimizer__seed*.jsonl \
  --out staging/h3/events.jsonl \
  --census-out staging/h3/event-census.json

conda run -n flowerfl python scripts/analyze_h3_identity.py \
  --events staging/h3/events.jsonl \
  --provenance staging/EXP-060/results/*__tge_fp__persistent_optimizer__seed*.json \
  --cohort adjudicating \
  --out staging/h3/verdict.json
```

H3 passes only if malicious rank-1 identity is at least 0.85 separately in S3
and S4, malicious wrong-device links are at most 0.10, honest wrong-device links
are at most 0.10, no adjudicating rank-1 event is indeterminate, and the event
census matches the design. The result was 80/80 rank-1 in each scenario, 0/160
malicious wrong-device links, and 0/15 honest wrong-device links. The one-sided
95% Clopper–Pearson limits were 0.9632458 below for each rank-1 rate, 0.0185491
above for the malicious guard, and 0.1810363 above for the honest guard.

## H4

The matrix is nine arms × C0/S0–S4 × the ten values in
[`seeds.json`](../../data/seeds.json) under `confirmatory_seeds`, 540 units. All
units use the sealed-test split and H2′ cuts v2. Use the 480-unit main and
60-unit arm-4 launch descriptors described above. Stage exactly one result JSON
per semantic cell and run the frozen scorer once to a new output path:

```bash
conda run -n flowerfl python scripts/analyze_h4_composition.py \
  --units staging/h4/results/*.json \
  --out staging/h4/H4_COMPOSITION_VERDICT.json \
  --memo staging/h4/H4_COMPOSITION_VERDICT.md
```

The primary treatment is H2P+FP+Krum and the comparator is Krum. For each seed
and S1–S4, each arm's degradation is its C0 `acc_final5` minus its scenario
`acc_final5`; the paired reduction is comparator degradation minus treatment
degradation. Every scenario needs median reduction ≥ 0.05 and exact paired
two-sided Wilcoxon p ≤ 0.05. S1 passed at +0.11560 and p = 0.005859. S2 failed
the median bar at −0.16459, S3 failed both at −0.03718 and p = 0.160156, and S4
failed the median bar at −0.08127. H4 was therefore falsified. Reported arms do
not alter that verdict.

H4′ is documented separately in [H4′ status](h4prime.md). It remains unrun and
its current descriptor cannot be rendered or launched.
