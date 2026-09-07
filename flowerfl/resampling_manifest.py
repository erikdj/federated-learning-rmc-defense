"""Durable per-client resampling manifest + arm-compliance assertions (§5).

Stage F records, for every client × unit, exactly what resampling did and what
the update-matching / weight semantics produced — as fit metrics that are NEVER
reused as aggregation weight. The manifest row (built by ``build_manifest_row``)
is the single source of truth the run record persists; ``assert_arm_compliance``
runs the LOUD checks §5 requires, raising ``ValueError`` (which FAILS the unit)
rather than letting a run finish while its recorded arm config disagrees with
what actually executed.
"""
from __future__ import annotations

from flowerfl.smote_resampler import semantic_attack_target_count

# Single source for the §5 weight-mode vocabulary — enforced identically at
# design-doc parse (praxis_exp.matrix_doc), container argv build
# (docker.entrypoint), and client run-config read (flowerfl.client_app), so no
# layer can accept a value another rejects.
WEIGHT_MODES: tuple[str, ...] = ("resampled", "original")


def validate_weight_mode(value) -> str:
    """Validate a ``weight_mode`` value, raising ``ValueError`` on anything
    outside ``WEIGHT_MODES`` (a typo like ``"orginal"`` must fail loudly, never
    silently run the incumbent weighting)."""
    if value not in WEIGHT_MODES:
        raise ValueError(
            f"weight_mode must be one of {list(WEIGHT_MODES)}, got {value!r}"
        )
    return value

# The full manifest schema (§5). ``prep_info`` supplies the resampling fields;
# the client supplies the identity / matching / weight fields.
MANIFEST_FIELDS: tuple[str, ...] = (
    "partition_id", "arm", "variant", "target_fraction",
    "n_orig", "n_resampled",
    "n_benign_before", "n_attack_before", "n_benign_after", "n_attack_after",
    "sampler_status", "skip_reason", "k_eff",
    "max_steps", "actual_steps",
    "weight_mode", "update_match", "num_examples", "semantic_policy",
)


def build_manifest_row(
    *,
    partition_id: int,
    arm: "str | None",
    weight_mode: str,
    update_match: bool,
    num_examples: int,
    max_steps: "int | None",
    actual_steps: "int | None",
    semantic_policy: bool,
    prep_info: dict,
) -> dict:
    """Assemble one manifest row from the load_data prep_info + client fields.

    ``prep_info`` carries n_orig / n_resampled / before+after class counts /
    sampler_status / skip_reason / k_eff / variant / target_fraction (§5); the
    client adds identity, the update-matching cap and its realized step count,
    the weight-mode and reported aggregation mass, and whether the semantic
    attack-class policy was in force.
    """
    row = {
        "partition_id": int(partition_id),
        "arm": arm,
        "max_steps": max_steps,
        "actual_steps": actual_steps,
        "weight_mode": weight_mode,
        "update_match": bool(update_match),
        "num_examples": int(num_examples),
        "semantic_policy": bool(semantic_policy),
    }
    row.update(prep_info)
    return row


def assert_arm_compliance(row: dict, *, expected: "dict | None" = None) -> None:
    """Loud §5 arm-compliance checks — raise ``ValueError`` (fails the unit).

    (1) update-matching: every client hit the cap EXACTLY (actual == max_steps);
    (2) weight-mode: reported num_examples == n_orig (original) / n_resampled
        (resampled);
    (3) the declared arm config ``(variant, target_fraction, weight_mode,
        update_match)`` matches the manifest (when ``expected`` is supplied);
    (4) the post-resampling class counts match the arm's semantic policy (§6).
    """
    pid = row.get("partition_id")

    # (1) update-matching hit the cap exactly.
    if row["update_match"]:
        if row["actual_steps"] != row["max_steps"]:
            raise ValueError(
                f"[MANIFEST] client {pid} arm {row['arm']}: update-match on but "
                f"actual_steps={row['actual_steps']} != max_steps={row['max_steps']}"
            )

    # (2) reported aggregation mass matches the weight mode.
    if row["weight_mode"] == "original":
        if row["num_examples"] != row["n_orig"]:
            raise ValueError(
                f"[MANIFEST] client {pid} arm {row['arm']}: weight-mode=original but "
                f"num_examples={row['num_examples']} != n_orig={row['n_orig']}"
            )
    elif row["weight_mode"] == "resampled":
        if row["num_examples"] != row["n_resampled"]:
            raise ValueError(
                f"[MANIFEST] client {pid} arm {row['arm']}: weight-mode=resampled but "
                f"num_examples={row['num_examples']} != n_resampled={row['n_resampled']}"
            )
    else:
        raise ValueError(
            f"[MANIFEST] client {pid} arm {row['arm']}: unknown weight_mode "
            f"{row['weight_mode']!r}"
        )

    # (3) declared arm config matches the manifest.
    if expected is not None:
        for k in ("variant", "target_fraction", "weight_mode", "update_match"):
            if row.get(k) != expected.get(k):
                raise ValueError(
                    f"[MANIFEST] client {pid} arm {row['arm']}: declared {k}="
                    f"{expected.get(k)!r} disagrees with manifest {row.get(k)!r}"
                )

    # (4) after-class-counts match the semantic policy.
    _assert_after_counts(row)


def _assert_after_counts(row: dict) -> None:
    """Post-resampling class counts must match the arm's declared mechanism (§6)."""
    pid, arm = row.get("partition_id"), row.get("arm")
    nb0, na0 = row["n_benign_before"], row["n_attack_before"]
    nb1, na1 = row["n_benign_after"], row["n_attack_after"]
    status = row["sampler_status"]

    if status in ("off", "skipped"):
        # No rebalancing ran — the split must be untouched.
        if (nb1, na1) != (nb0, na0):
            raise ValueError(
                f"[MANIFEST] client {pid} arm {arm}: sampler_status={status} but "
                f"after-counts ({nb1},{na1}) != before-counts ({nb0},{na0})"
            )
        return

    if status != "applied":
        raise ValueError(
            f"[MANIFEST] client {pid} arm {arm}: unknown sampler_status {status!r}"
        )

    if not row["semantic_policy"]:
        # Legacy min/max path carries no semantic contract; nothing to enforce
        # beyond the applied status.
        return

    f = _target_fraction(row["target_fraction"])
    variant = row["variant"]
    if variant in ("smote", "random_over"):
        # Over-samplers hold benign fixed and grow attack to the ceil target.
        expected_attack = semantic_attack_target_count(f, nb0)
        if nb1 != nb0 or na1 != expected_attack:
            raise ValueError(
                f"[MANIFEST] client {pid} arm {arm}: semantic over-sampler expected "
                f"benign={nb0}, attack={expected_attack}; got benign={nb1}, attack={na1}"
            )
    elif variant == "random_under":
        # Under-sampler holds attack fixed and shrinks benign to the f fraction.
        if na1 != na0:
            raise ValueError(
                f"[MANIFEST] client {pid} arm {arm}: semantic under-sampler grew/shrank "
                f"attack ({na0} -> {na1}); attack must be untouched"
            )
        realized = na1 / (na1 + nb1) if (na1 + nb1) else 0.0
        if realized < f - 1e-9:
            raise ValueError(
                f"[MANIFEST] client {pid} arm {arm}: semantic under-sampler realized "
                f"attack fraction {realized:.4f} < target f={f}"
            )
    else:
        raise ValueError(
            f"[MANIFEST] client {pid} arm {arm}: unknown variant {variant!r}"
        )


def _target_fraction(target) -> float:
    """Resolve the manifest ``target_fraction`` to a float f (``balanced`` == 0.5)."""
    if target in ("balanced", None):
        return 0.5
    return float(target)


def assert_manifest_complete(
    rows_by_pid: "dict[int, dict]",
    expected_partitions: "set[int]",
    *,
    unresolved_cids: "set[str] | None" = None,
) -> None:
    """Unit-level completeness gate (§5: one row per client per unit).

    Given the collected manifest rows keyed by ``partition_id`` and the
    authoritative set of partitions the strategy dispatched for fit, raise
    ``ValueError`` (which FAILS the unit) if the collected rows do not EXACTLY
    cover the dispatched set with schema-complete rows. Four violation classes
    are reported together so an operator sees the full picture:

    * partitions dispatched but MISSING a manifest row (evidence loss — a
      post-dispatch client failure that vanished from a partial manifest);
    * manifest rows for partitions that were NEVER dispatched (a row that
      cannot be reconciled against the scenario);
    * rows MISSING any key from ``MANIFEST_FIELDS`` (schema-incomplete row);
    * ``unresolved_cids`` — clients dispatched during the discovery round that
      NEVER resolved to a partition (a discovery-round failure). Without this
      class such a client is absent from BOTH the collected manifest and the
      expected partition set, so the first three checks pass while the unit
      still lost that client's evidence. ``None``/empty is a no-op.

    "Nonempty" is never sufficient — an empty collection against a nonempty
    dispatched set is itself a missing-coverage violation.
    """
    collected = set(rows_by_pid)
    missing = expected_partitions - collected
    unexpected = collected - expected_partitions
    unresolved = set(unresolved_cids or ())

    schema_violations: "dict[int, list[str]]" = {}
    for pid, row in rows_by_pid.items():
        absent = [k for k in MANIFEST_FIELDS if k not in row]
        if absent:
            schema_violations[pid] = absent

    if not (missing or unexpected or schema_violations or unresolved):
        return

    parts: list[str] = []
    if missing:
        parts.append(
            f"partitions dispatched but missing a manifest row: {sorted(missing)}"
        )
    if unexpected:
        parts.append(
            f"manifest rows for partitions never dispatched: {sorted(unexpected)}"
        )
    if schema_violations:
        detail = "; ".join(
            f"partition {pid} missing {keys}"
            for pid, keys in sorted(schema_violations.items())
        )
        parts.append(f"rows missing required MANIFEST_FIELDS keys: {detail}")
    if unresolved:
        parts.append(
            f"clients dispatched during discovery that never resolved to a "
            f"partition: {sorted(unresolved)} — a discovery-round failure "
            f"would otherwise vanish from the evidence"
        )

    raise ValueError(
        "[MANIFEST] incomplete per-client resampling manifest — " + "; ".join(parts)
    )
