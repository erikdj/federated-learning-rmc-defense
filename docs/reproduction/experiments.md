# Experiments and analysis

All historical and fresh reproduction matrices use `persistent_optimizer`, a
2,000,000-sample per-client cap, 50 rounds, and one repeat. Scenario behavior
comes from the JSON files in [`rmc/scenarios`](../../rmc/scenarios); filenames
are identifiers and do not override their contents.

S1 combines nine sustained ALIE attackers with honest churn. S3 applies
Gaussian noise in rounds 3–7 from the original malicious identities, then ALIE
from each reset identity; its first reset therefore also changes attack family.
Its malicious return rounds are 9, 19, 29 and 39. S4 combines honest churn,
identity resets and the Gaussian→ALIE→label-flip→Gaussian→ALIE attack sequence;
its malicious return rounds are 8, 20, 32 and 42. These schedules are fixed but
differ across scenarios, so scenario names alone do not define a contrast that
changes only one factor. C0 is the no-attack reference.

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
  --image-digest "$IMAGE_DIGEST" --no-push
```

This example keeps the launch's Git tag locally and requires no push access to
the upstream repository. `--no-push` still submits the AWS Batch experiment;
it only suppresses Git pushes. When working in your own fork, omit it to push
the committed launch and tags to the current branch, or use `--branch NAME`
to choose the remote destination. The same options apply to `--refill`.

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
`[42, 137, 256, 314, 500]`, 25 cells, with train-only client normalization. Historically
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
subset of EXP-048: S0–S4 × ten seeds, 50 signal logs. The public
[threshold artifact](../../reproduction/evidence/h1-thresholds-dev-frozen.json)
records one development-frozen cut per family:
S = 0.6336447001006367, W = 0.8974294186230957 and
C = 0.7016743007511332. The cut is the linear 0.90 quantile of 915 honest
development scores; a malicious-risk score is flagged only when it is strictly
greater than the cut. Apply those cuts unchanged to the held-out corpus.
Stage the 25 development logs and 50 held-out logs in separate directories.
The scorer freezes cuts from the development logs before scoring the held-out
logs and writes the read and thresholds into a new output directory:

```bash
conda run -n flowerfl python reproduction/analyze_h1.py \
  --dev-signals staging/h1-training/signals \
  --heldout-signals staging/EXP-048/signals \
  --models models/h1_signal_family/leakfree_25cell \
  --out staging/h1-recomputed
```

The output directory contains `h1_corrected_read.json`,
`thresholds_dev_frozen.json`, and `READ.md`. The public evidence distributes
the read as `h1-heldout.json` and the cuts as `h1-thresholds-dev-frozen.json`.

The three conjunctive criteria are C−S ≥ 0.05 in each of S2–S4, C−W ≥ 0.05 in
each of S2–S4, and one-sided paired exact Wilcoxon p ≤ 1/32 for both comparisons
in each scenario. H1 was not met: the C−S margins were −0.0407, +0.0007 and
−0.0867; the C−W margins were +0.0556, +0.0200 and −0.1874. The combined family
never gains 0.05 over the better single family in S2–S4. Across S0–S4 and the
three families, realized FPR averaged over seeds ranges from 0.0515 to 0.1121;
individual seed cells vary more. The evidence also reports a separate
per-cell matched-operating-point sensitivity. The development-frozen-cut read
is the primary result.

## H2

EXP-048 is four defenses (`Krum`, `TrustScore`, standalone `TGE`, `Krum+TGE`) ×
S0–S4 × seeds `[1009, 1733, 2521, 3299, 4127, 5051, 6079, 7177, 8231, 9337]`,
200 units, with train-only client normalization. The all-rounds scorer uses frozen
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

These are comparisons at frozen cuts, not comparisons matched on realized FPR.
Mean all-round FPR is 0.159 for TGE versus 0.044 for Krum in S2, 0.414 for TGE
versus 0.005 for Krum+TGE in S3, and 0.062 for TGE versus 0.104 for Krum in S4.
Each defense's cut stays fixed across the scenarios; the nominal calibration
budget does not make the realized operating points equal.

<a id="h2prime"></a>

## H2′

The H2′ corpus is Krum+TGE × S0–S4 × the ten values in
[`h2prime_confirm_seeds.json`](../../data/h2prime_confirm_seeds.json), 50 signal
logs. The confirmatory read uses a fixed rotation over these ten seeds. After
sorting them ascending, test seed i uses the next cyclic seed for honest
calibration and the following three cyclic seeds for fitting. A seed can thus
serve different roles in different rotations; the read does not evaluate one
classifier fitted once on the development corpus. One GBDT is fit per rotation
and held-out attack family, excluding fit rows with that current attack label. Each
scenario uses the linear 0.90 quantile of honest calibration scores and flags
only scores strictly greater than the cut. NumPy 2.2.6 and scikit-learn 1.7.2
are protocol pins.

Window features are derived before the current-label exclusion. In S2, the two
rows after a strategy switch can therefore carry statistics from the family
withheld from the fit. This distinction limits the primary claim to exclusion
by current attack label; the window-aware sensitivity below also excludes fit
rows whose derived window contains the withheld family.

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

P1 weights each family's recall by its malicious-row count within a scenario
cell, then averages the resulting S4 blended recall over the ten test seeds.
The corresponding blended FPR uses the same family weights. P1 requires mean
blended leave-one-attack-out S4 recall ≥ 0.35 with the mean
realized blended FPR in `[0.08, 0.12]`. It passed at recall 0.502020202020202 and
FPR 0.1026039919490104. P2 takes the equal-weight mean of the five ALIE-bearing
scenario-cell recalls within each seed. Its baseline first takes the oracle
maximum of Krum score, L2-to-median and cosine-to-median in each cell, then
averages those five values. The detector must strictly beat that baseline in at
least nine of ten seeds, with detector and selected-baseline pooled FPR both in the same
interval. It passed 10/10; exact sign p was 0.0009765625, detector FPR was
0.11551155115511551, and baseline FPR was 0.10843417675100843. H2′ is confirmed
because both components passed.

All three oracle baselines have zero ALIE recall in all 50 P2 cells. Krum's
selection in 50/50 cells is the fixed-order tie-break among those zero recalls,
not evidence that Krum outperforms the other two baseline scores.

The `secondaries.window_aware_loao_sensitivity` block repeats fitting after
excluding rows whose derived window contains the withheld family. Its corpus
census finds 540 malicious S2 rows carrying another family's history, and none
in S0, S1, S3 or S4. The sensitivity gives S4 recall 0.5237 at FPR 0.1037,
compared with primary recall 0.5020 at FPR 0.1026. Detector superiority remains
10/10 seeds in both reads. Scenario-mean blended-recall changes range from
−0.0708 to +0.0567. These are reported sensitivity quantities; the primary P1
and P2 values and verdict remain unchanged.

## H3

EXP-059 is the non-adjudicating corrected-instrument validation matrix: TGE and
TGE+FP × S3/S4 × the five public development seeds, 20 units, using
`identity_only`, validation cohort, and τ = 18.639429816855873. Its integrity
gates opened EXP-060. EXP-060 repeats the same arms and scenarios on seeds
`[11749, 19025, 66912, 75901, 93207]`, using adjudicating cohort,
`identity_only`, τ = 26.466874982783164, and scoring on odd-numbered partitions.

Only the final threshold τ and covariance Σ are fitted on the even-numbered
partitions. R5 feature screening reads all 20 partitions, and a pre-lock
homogeneity gate checks the odd partitions that will be scored. The screen
uses device-grouped distribution and variability measures from controls and
raw pools, not held-out attack labels. The test uses fresh seeds, but its
scored partitions are not unseen during instrument construction and acceptance.

Each round, a client computes its fingerprint from a fresh 100,000-row bootstrap sample
with replacement from a fixed raw training pool of at most 100,000 rows. Four
moments of the 45 columns produce the 180-dimensional fingerprint; the metric uses the locked
58-dimensional mask. An isolated RNG keyed by partition, seed and round keeps
fingerprint sampling separate from training. Re-entry retains the underlying
partition, so this is simulated partition-based identity linkage, not a test of
physical-device authentication or changing network conditions. The identity-only
registry consults previously enrolled profiles independently of detector flags;
its linkage result does not establish enforcement effectiveness.

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
The pass rule uses point estimates, not these confidence bounds. The 15 honest
events comprise three reconnects across five seeds; the upper bound above 0.10
records their limited precision. Rank-1 identity is threshold-free, while the
wrong-device guards count links made at the locked τ.

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

Client loading uses training-only normalization. The server's sealed evaluator
separately computes feature means and standard deviations from the concatenated
held-out evaluation matrix, then z-scores that matrix. This label-free transform
uses the same construction for every arm; it does not apply the clients'
training-fitted moments. Shared evaluation preprocessing does not establish
unbiased paired contrasts. The frozen row-index split also does not establish
row-disjointness from local training data, so the utility findings remain
conditional on these evaluation limits.

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
