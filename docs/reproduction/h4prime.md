# H4′ status and future protocol

H4′ has no result. The source design is dated 2026-08-26 and remains explicitly
draft, awaiting ratification. No EXP-066 fleet, scorer, fresh seed file, or
terminal verdict exists. The public descriptor
[`h4prime-draft-no-krum.json`](../../reproduction/configs/h4prime-draft-no-krum.json)
is marked `launchable: false` so it cannot be mistaken for a preregistered run.

The proposed public core contains 120 units: `H2P+FP` and `FedAvg`, six scenario
files (`C0`, `S0`–`S4`), ten future sealed seeds, persistent optimizer, a
2,000,000-sample client cap, and 50 rounds. It uses the sealed-test evaluator,
H2′ cuts v2, and the validation `flag_gated` fingerprint instrument. Krum is
excluded from this public design.

The proposed endpoint follows the H4 structure. For each seed and scenario,
compute `acc_final5(C0) - acc_final5(SX)` within each arm, then subtract the
H2P+FP degradation from the FedAvg degradation. Each of S1–S4 would need a
median reduction of at least 0.05 and an exact paired two-sided Wilcoxon
p-value at most 0.05; all four would have to pass both bars.

The scenario filenames are historical labels. Their JSON contents are the
authority for scheduled behavior. In particular,
[`S1_benign_churn_only.json`](../../rmc/scenarios/S1_benign_churn_only.json)
schedules sustained ALIE attackers as well as churn, so S1 must not be
described as an attack-free benign condition.

Before this can become a confirmatory experiment, a dated ratified revision
must freeze and review the scorer and deterministic seed-draw implementation,
draw one fresh set of ten seeds disjoint from every prior cohort, pin the seed
file in the scorer, and replace the empty seed array in a new launchable config.
Results from the already exposed H4 seeds cannot adjudicate H4′.

