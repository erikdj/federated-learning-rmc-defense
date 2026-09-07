"""Flag-gated per-client oversampling for the FL training split (GWU-59).

SMOTE study knob. This module is a pure, deterministic resampler applied to a
single client's TRAINING split only (see flowerfl/task.py::load_data), after
partitioning and the per-client example cap and before model training. It is
inert unless a run explicitly enables it via the ``smote-enabled`` run-config
key (default OFF everywhere — pyproject, runner, provenance).

Design notes:
  - imbalanced-learn is imported LAZILY inside resample_training_split so the
    incumbent (SMOTE-off) code path never pays the import and the FL image stays
    lean when the knob is unused.
  - Determinism is seeded from the caller's derive_seed machinery (per
    (base_seed, client) — load_data runs once per client per run, before the
    server round is known), so the same inputs + seed yield byte-identical
    resampled output.
  - Validation is LOUD: an unknown variant or an out-of-range target raises
    ValueError at config-parse time (flowerfl/client_app.py), never silently
    downgrading to a different behaviour.
"""
from __future__ import annotations

import math

import numpy as np

# Extensible registry of supported oversampling variants. Each maps to a factory
# that builds the imbalanced-learn sampler for a resolved sampling_strategy +
# random_state. Add new variants here (e.g. "adasyn", "borderline") — the
# validation and provenance surfaces pick them up automatically.
SUPPORTED_SMOTE_VARIANTS: tuple[str, ...] = ("smote", "random_over", "random_under")

# Over-samplers GROW the minority to hit the target ratio (originals preserved,
# majority untouched). The lone under-sampler SHRINKS the majority to hit the
# SAME ratio (minority untouched, majority rows removed). Kept for the variant
# dispatch + provenance semantics below.
_OVER_SAMPLING_VARIANTS: frozenset[str] = frozenset({"smote", "random_over"})
_UNDER_SAMPLING_VARIANTS: frozenset[str] = frozenset({"random_under"})

# User-facing target vocabulary; "balanced" == 50/50 per client.
_BALANCED = "balanced"


def validate_smote_variant(variant) -> str:
    """Return ``variant`` if supported, else raise ValueError loudly."""
    if variant not in SUPPORTED_SMOTE_VARIANTS:
        raise ValueError(
            f"smote_variant must be one of {SUPPORTED_SMOTE_VARIANTS}, got {variant!r}"
        )
    return variant


def coerce_smote_enabled(value) -> bool:
    """Strictly coerce a ``smote_enabled`` value to bool, raising on anything
    unrecognized.

    Accepts a real bool or the exact (case-insensitive, surrounding-whitespace-
    tolerant) string tokens ``"true"``/``"false"``. Everything else — a typo like
    ``"ture"``, a near-miss like ``"yes"``, a number like ``1``/``1.5``, None —
    raises ValueError. The old behaviour silently coerced any unrecognized string
    to False, which would run an entire registered SMOTE arm as the incumbent.
    Single source of truth for both the pre-registration parse (matrix_doc) and
    the container defense-in-depth (entrypoint)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "true":
            return True
        if token == "false":
            return False
    raise ValueError(
        f"smote_enabled must be a boolean or the string 'true'/'false', got {value!r}"
    )


def normalize_smote_target(target) -> str | float:
    """Normalize a target into the canonical vocabulary or raise loudly.

    Accepts the string ``"balanced"`` (50/50) or a minority-ratio float in
    (0, 1] (possibly given as a numeric string). Booleans and out-of-range or
    non-numeric values raise ValueError.
    """
    if isinstance(target, str) and target == _BALANCED:
        return _BALANCED
    # bool is an int subclass — reject it explicitly so True/False can't slip
    # through as 1.0/0.0.
    if isinstance(target, bool):
        raise ValueError(f"smote_target must be 'balanced' or a float in (0, 1], got {target!r}")
    try:
        value = float(target)
    except (TypeError, ValueError):
        raise ValueError(
            f"smote_target must be 'balanced' or a float in (0, 1], got {target!r}"
        )
    if not (0.0 < value <= 1.0):
        raise ValueError(
            f"smote_target float must be in (0, 1], got {value!r}"
        )
    return value


def effective_k_neighbors(y: np.ndarray) -> int:
    """The SMOTE ``k_neighbors`` actually used for label vector ``y``.

    Single source of truth for the small-partition clamp: k_neighbors must be
    < the minority class count, so it is capped at min(5, minority_count − 1)
    (floored at 1). Only meaningful when the minority has >= 2 samples (the
    n_minority < 2 case is a starvation skip, handled in resample_training_split);
    reused by flowerfl/task.py::load_data to report the applied k in the
    per-client provenance record so the two can never drift.
    """
    counts = np.bincount(y.astype(int))
    present = counts[counts > 0]
    n_minority = int(present.min()) if present.size else 0
    return max(1, min(5, n_minority - 1))


def _sampling_strategy(target: str | float) -> str | float:
    """Map the canonical target to an imbalanced-learn ``sampling_strategy``.

    "balanced" -> "auto"; a float passes through as the desired minority/majority
    ratio AFTER resampling. imbalanced-learn's ``sampling_strategy`` uses the
    identical float definition for over- and under-samplers (minority-over-
    majority count post-resample), so the same value is coherent for both — only
    the MECHANISM differs: an over-sampler grows the minority up to the ratio, an
    under-sampler shrinks the majority down to it. For "auto"/"balanced":
    over-samplers grow every non-majority class up to the majority count (50/50),
    under-samplers shrink every non-minority class down to the minority count
    (also 50/50). Binary in both cases.
    """
    return "auto" if target == _BALANCED else float(target)


# Starvation reasons recorded in provenance when oversampling is skipped rather
# than attempted (DESIGN.md §6c). A skip is the DESIGNED outcome for a client
# whose training split cannot be meaningfully oversampled — never a crash: a
# crash at client data prep on Batch is a lost 4-5h unit, a flagged skip is not.
SKIP_SINGLE_CLASS = "single_class"      # only one class present in the split
SKIP_MINORITY_STARVED = "minority_starved"  # minority class has < 2 samples (over-samplers)
# Under-sampling can't "starve" the minority (it is preserved), but a requested
# ratio AT OR BELOW the observed minority/majority ratio is infeasible or a
# no-op: the under-sampler can only RAISE the ratio by removing majority rows, so
# a target <= observed either requires growing the majority (impossible) or
# changes nothing (a rounded majority-target can hide this). Skip-with-provenance,
# distinct reason — a no-op must never report "applied" (P2-2).
SKIP_UNDERSAMPLE_DEGENERATE = "undersample_degenerate"
# Semantic attack-class policy (Stage-F §6): the over-samplers grow the ATTACK
# class (label 1); a client whose attack fraction is ALREADY at/above the target
# floor f (attack-dominant OR exactly at target) is a no-op — benign is NEVER
# grown, so the split is left untouched with this reason rather than crashing.
SKIP_ATTACK_AT_OR_ABOVE_TARGET = "attack_at_or_above_target"


def semantic_attack_target_count(f: float, n_benign: int) -> int:
    """Post-resampling attack-class (label 1) count for a target fraction ``f``.

    Stage-F §6 semantic policy: hold the benign class (label 0) fixed at
    ``n_benign`` and grow the attack class so the attack FRACTION reaches at
    LEAST ``f``. Solving ``n_a / (n_a + n_benign) >= f`` for ``n_a`` gives
    ``n_a >= f/(1-f) * n_benign``; ``math.ceil`` (never ``round``) is what makes
    the realized fraction ``>= f`` — the ``round`` counterexample ``f=.47,
    n_benign=5`` rounds down to ``4/9 = .444 < .47`` while ``ceil`` gives
    ``5/10 = .50``. Proof and a 10^5-case exhaustive check are in DESIGN §6.

    ``f`` must be a proper fraction in (0, 1); ``f >= 1`` is not a rebalancing
    target (it would divide by zero / demand an all-attack split) and raises.
    """
    if not (0.0 < f < 1.0):
        raise ValueError(
            f"semantic attack-fraction target f must be in (0, 1), got {f!r}"
        )
    return math.ceil(f / (1.0 - f) * n_benign)


def validate_semantic_target_combo(target) -> None:
    """Reject a semantic-policy target outside (0, 1) BEFORE any run exists.

    The legacy min/max ratio vocabulary allows ``smote_target`` in (0, 1] —
    but under the semantic attack-class policy (Stage-F §6) a unit target is
    fatal in-run: over-samplers raise in ``semantic_attack_target_count``
    (f/(1−f) diverges) and the under-sampler computes a ZERO benign target.
    Single source called at design-doc parse (matrix_doc) and container argv
    build (entrypoint), so the combination fails at pre-registration, never
    mid-run on the fleet.
    """
    f = 0.5 if target in (_BALANCED, None) else float(target)
    if not (0.0 < f < 1.0):
        raise ValueError(
            f"smote_semantic_target requires a target fraction in (0, 1); "
            f"got smote_target={target!r} (legal for the legacy min/max "
            f"vocabulary, fatal for the semantic attack-class policy)"
        )


def resample_training_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    variant: str,
    target,
    seed: int,
    attack_target_policy: bool = False,
) -> tuple[np.ndarray, np.ndarray, "str | None"]:
    """Oversample the (X, y) training split deterministically.

    Args:
        X: 2-D float feature matrix (n_samples, n_features).
        y: 1-D integer label vector (n_samples,).
        variant: one of SUPPORTED_SMOTE_VARIANTS.
        target: "balanced" or a minority-ratio float in (0, 1].
        seed: RNG seed for reproducible synthesis.

    Returns:
        (X_res, y_res, skipped_reason). On the normal path skipped_reason is None.
        For the OVER-samplers ("smote", "random_over") the minority is grown to
        the requested ratio (originals preserved; majority untouched). For the
        UNDER-sampler ("random_under") the majority is shrunk to the requested
        ratio (minority preserved; majority rows removed) — no synthesis. When
        the split cannot be resampled the input is returned UNCHANGED with a
        non-None reason (SKIP_SINGLE_CLASS shared; SKIP_MINORITY_STARVED for the
        over-samplers; SKIP_UNDERSAMPLE_DEGENERATE for random_under) — a
        skip-with-provenance, never an exception.

    Semantic asymmetry (documented deliberately): over-samplers ADD rows to a
    minority with >= 2 samples, so a 1-sample minority is un-oversamplable and
    starves. The under-sampler REMOVES majority rows and preserves the minority,
    so a 1-sample minority is valid (majority shrinks to 1). Its degenerate cases
    are instead a class dropping below 1 row or a float target below the observed
    data ratio (which would require growing the majority).
    """
    variant = validate_smote_variant(variant)
    canonical_target = normalize_smote_target(target)
    strategy = _sampling_strategy(canonical_target)
    random_state = int(seed) % (2**32)

    # Single-class guard is shared by all variants — evaluated BEFORE building any
    # sampler so imblearn can never raise inside fit_resample on a degenerate split.
    counts = np.bincount(y.astype(int))
    present = counts[counts > 0]
    if present.size < 2:
        return X, y, SKIP_SINGLE_CLASS
    n_minority = int(present.min())
    n_majority = int(present.max())

    # Stage-F §6 SEMANTIC attack-class policy (opt-in). The legacy path below is
    # label-AGNOSTIC (grows/shrinks whichever class is the numeric minority/
    # majority); the semantic path instead grows/shrinks by LABEL IDENTITY —
    # attack = label 1, benign = label 0 — so an attack-dominant client no-ops
    # rather than oversampling benign. Kept opt-in so the incumbent path stays
    # byte-for-byte the min/max behaviour the Stage-C/D/E arms were run under.
    if attack_target_policy:
        return _resample_semantic(
            X, y, variant=variant, target=canonical_target, random_state=random_state
        )

    # Lazy import: keeps the SMOTE-off path free of imbalanced-learn.
    if variant in _UNDER_SAMPLING_VARIANTS:  # "random_under"
        # Feasibility guard: the under-sampler can only
        # RAISE minority/majority by REMOVING majority rows. Any requested ratio
        # <= the OBSERVED ratio (n_minority/n_majority) is infeasible or a no-op.
        # Compare the ratio DIRECTLY — a rounded majority-target hides the no-op
        # (n_min=2, n_maj=3, target 0.6 -> round(2/0.6)=3 == current majority, so
        # nothing changes while status would say "applied").
        observed_ratio = n_minority / n_majority
        if canonical_target == _BALANCED:
            # "balanced" == target ratio 1.0; a no-op iff already balanced.
            if observed_ratio >= 1.0:
                return X, y, SKIP_UNDERSAMPLE_DEGENERATE
        else:
            if float(canonical_target) <= observed_ratio:
                return X, y, SKIP_UNDERSAMPLE_DEGENERATE

        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
    else:  # over-samplers share the 1-sample-minority starvation policy
        if n_minority < 2:
            return X, y, SKIP_MINORITY_STARVED
        if variant == "random_over":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
        else:  # "smote"
            from imblearn.over_sampling import SMOTE

            # k_neighbors must be < the minority class count; the shared clamp
            # (n_minority == 2 -> k == 1) keeps the value load_data reports in the
            # per-client record identical to the value SMOTE actually uses. The
            # n_minority < 2 case is already handled above.
            sampler = SMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=effective_k_neighbors(y),
            )

    X_res, y_res = sampler.fit_resample(X, y)
    return np.asarray(X_res, dtype=X.dtype), np.asarray(y_res, dtype=y.dtype), None


def _resample_semantic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    variant: str,
    target: "str | float",
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, "str | None"]:
    """Stage-F §6 semantic attack-class (label 1) resampling.

    ``target`` is the attack-fraction floor ``f`` — ``"balanced"`` == f=0.50, a
    float passes through as f directly. Over-samplers (smote, random_over) GROW
    the attack class to ``semantic_attack_target_count(f, n_benign)`` with benign
    held fixed, applied ONLY when that target strictly exceeds the current attack
    count; the under-sampler (random_under) SHRINKS benign until the attack
    fraction reaches f (f=0.50 ⇒ benign == n_attack). The four no-op guards of §6
    return the split UNCHANGED with a provenance reason — benign is NEVER grown
    and nothing ever raises (single-class is already handled by the caller).
    """
    f = 0.50 if target == _BALANCED else float(target)
    n_benign = int((y == 0).sum())
    n_attack = int((y == 1).sum())

    if variant in _UNDER_SAMPLING_VARIANTS:  # "random_under": shrink benign
        # Target benign count so the attack fraction reaches f (benign = attack ·
        # (1-f)/f; f=0.50 ⇒ benign == n_attack). A no-op iff attack is already at/
        # above the target fraction — the under-sampler can only RAISE the attack
        # fraction by removing benign rows, never lower it.
        benign_target = int(round((1.0 - f) / f * n_attack))
        if benign_target >= n_benign:
            return X, y, SKIP_UNDERSAMPLE_DEGENERATE
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(
            sampling_strategy={0: benign_target}, random_state=random_state
        )
    else:  # over-samplers: grow attack, benign held fixed
        n_attack_target = semantic_attack_target_count(f, n_benign)
        if n_attack_target <= n_attack:
            # Attack fraction already at/above f (attack-dominant or at-target):
            # never grow benign — no-op with provenance.
            return X, y, SKIP_ATTACK_AT_OR_ABOVE_TARGET
        if variant == "smote" and n_attack < 2:
            # SMOTE k-NN interpolation needs >= 2 attack rows; random_over can
            # duplicate a single row, so only smote starves here (§6).
            return X, y, SKIP_MINORITY_STARVED
        if variant == "random_over":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(
                sampling_strategy={1: n_attack_target}, random_state=random_state
            )
        else:  # "smote"
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(
                sampling_strategy={1: n_attack_target},
                random_state=random_state,
                k_neighbors=effective_k_neighbors(y),
            )

    X_res, y_res = sampler.fit_resample(X, y)
    return np.asarray(X_res, dtype=X.dtype), np.asarray(y_res, dtype=y.dtype), None
