#!/usr/bin/env python3
"""Frozen computation core for the H4 composition scorer.

Design authority (binding, in order):

* The H4 workflow in `docs/reproduction/experiments.md`, ratified as
  methodology v1.51;
* historical methodology v1.52 — E1 (C0 reference), E2 (arm 9 `h2p_ts`),
  E3/E3-bis, E4 (sealed-test custody), E5 (scorer freeze point);
* `docs/reproduction/experiments.md` (interfaces).

This module holds the pre-registered constants and the pure arithmetic:
endpoints (`acc_final5`, single-final-round, F1 analog), the exact paired
two-sided Wilcoxon signed-rank test, the § 4.1 gate, and the § 6
reference-anomaly window. Everything discretionary was frozen upstream;
nothing here is tunable at run time.

Loading, custody corroboration, census gating and report assembly live in
`scripts/analyze_h4_composition.py` (the CLI), mirroring the
`analyze_h3_identity.py` fail-closed architecture.

SEED DISCIPLINE (stricter than the H3 scorer): the sealed confirmatory
seeds are consumed by H4 for the first time, so NO seed value may appear
in any log, refusal, or output JSON. Units are identified in refusals by
arm/scenario plus SHA-256 prefixes of their path (launch tooling embeds
the seed in filenames) via `redacted`; seeds appear in outputs only as
`seed_ordinal` — the index into the sorted sealed manifest.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ScoringError(Exception):
    """A refusal. The scorer never scores past one of these."""


# ===========================================================================
# Pre-registered constants (FROZEN — v1.51 as amended by erratum A, v1.52)
# ===========================================================================

STATUS = (
    "FROZEN H4 scorer — pre-registration 2026-08-16 (RATIFIED, methodology "
    "v1.51) as amended by erratum A 2026-08-17 (RULED, v1.52). The ONE "
    "execution on the completed 9x6x10 census adjudicates H4; the primary "
    "contrast is arm 1 (h2p_fp_krum) vs arm 2 (krum) and no other arm can "
    "rescue, overturn, soften or strengthen it."
)

#: sha256 of `data/h4_serving/manifest.json` bytes — the Lane-A § 7.1 serving
#: bundle's v1 `bundle_sha256` (BUILD_CONTRACT: "sha256 of manifest.json
#: bytes"). RETAINED for interpreting EXP-061 artifacts ONLY: erratum B
#: (2026-08-18, methodology v1.53) retired the v1 corpus-quantile cuts after
#: the EXP-061 C3 non-transfer — this value is NOT accepted for the sealed
#: fleet.
SERVING_BUNDLE_SHA256_V1 = (
    "56eca8e31d1352a73e70962d2d3375e412def6f983bae6be29a7897fef16db8a"
)

#: The ACTIVE custody pin for the sealed fleet: sha256 of
#: `data/h4_serving/manifest_v2.json` bytes (bundle_v2_sha256). Pinned
#: 2026-08-20 in the SAME commit that lands the bundle-v2 files, from the
#: EXP-062 36/36 census build at source commit fd37fa4 (erratum B §§ B1-B2
#: as amended by erratum C § C1; two disclosed tie-fallback cells:
#: (C0, ts_family) realized 0.1095 and (S1, fedavg_family) 0.1009).
SERVING_BUNDLE_SHA256 = (
    "3bfeefb45700dff2e02a14acb0de4acfadcc717a7b82d6c246d44a8f44fa062c"
)


def required_serving_bundle_sha256() -> str:
    """The sealed fleet's serving-bundle pin, or a refusal.

    Erratum B (2026-08-18, RULED, methodology v1.53): the custody gate moves
    from the v1 pin to bundle_v2_sha256. This accessor is the single gate —
    while the pin is the `TBD_BUNDLE_V2` sentinel (bundle v2 not yet built)
    or anything that is not a 64-hex sha256, every scoring path that needs
    the pin refuses loudly instead of comparing against a placeholder.
    """
    value = str(SERVING_BUNDLE_SHA256)
    if value == "TBD_BUNDLE_V2":
        raise ScoringError(
            "serving-bundle pin is the TBD_BUNDLE_V2 sentinel: erratum B "
            "(docs/superpowers/specs/2026-08-18-h4-preregistration-erratum-b"
            ".md, methodology v1.53) retired the v1 cuts for the sealed "
            "fleet, and bundle v2 has not been pinned yet. Build bundle v2 "
            "(scripts/build_h4_serving_cuts_v2.py) and pin its "
            "bundle_v2_sha256 here before scoring. The v1 pin "
            "(SERVING_BUNDLE_SHA256_V1) interprets EXP-061 artifacts only."
        )
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ScoringError(
            f"serving-bundle pin {value!r} is not a 64-hex sha256 — a "
            "malformed pin must never become the sealed fleet's custody bar "
            "(erratum B pin-swap discipline)."
        )
    return value

#: The sealed evaluation split (erratum-A E4): H4 units evaluate on the test
#: indices of this committed manifest, locked 2026-05-14, never re-cut.
SPLIT_MANIFEST_PATH: Path = PROJECT_ROOT / "data" / "val_test_split_manifest.json"
REQUIRED_EVAL_SPLIT = "sealed_test"

#: The sealed seeds — base spec § 6.4; H4 is their first and only consumer.
SEED_MANIFEST_PATH: Path = PROJECT_ROOT / "data" / "seeds.json"
SEED_MANIFEST_KEY = "confirmatory_seeds"
N_SEEDS = 10

#: § 2 as amended by erratum-A E2: the NINE arms, by pre-registered number.
ARM_BY_NUMBER: Dict[int, str] = {
    1: "h2p_fp_krum",   # ADJUDICATING treatment — never drops
    2: "krum",          # ADJUDICATING comparator — never drops
    3: "h2p_fp",
    4: "krum_tge_fp",
    5: "h2p_krum",
    6: "h2p_fp_ts",
    7: "trustscore",
    8: "fedavg",
    9: "h2p_ts",
}
ARM_TOKENS = frozenset(ARM_BY_NUMBER.values())
TREATMENT_ARM = ARM_BY_NUMBER[1]
COMPARATOR_ARM = ARM_BY_NUMBER[2]

#: Arms serving the frozen § 7.1 H2′ bundle (custody: exact bundle sha).
#: `krum_tge_fp` detects with TGE, not the H2′ bundle — its sha is null.
DETECTOR_ARMS = frozenset({
    "h2p_fp_krum", "h2p_fp", "h2p_krum", "h2p_fp_ts", "h2p_ts",
})
#: Arms running the FLAG_GATED identity layer (custody: fp_registry_policy).
FP_ARMS = frozenset({"h2p_fp_krum", "h2p_fp", "h2p_fp_ts", "krum_tge_fp"})

#: Erratum-A drop order by arm number: 8 -> 4 -> 7 -> 9 -> 6 -> 5 -> 3.
#: Dropped arms must form a PREFIX of this sequence, all-or-nothing per arm;
#: arms 1 and 2 never drop; C0 cells are required for every surviving arm.
DROP_ORDER = tuple(ARM_BY_NUMBER[n] for n in (8, 4, 7, 9, 6, 5, 3))
NEVER_DROP = (TREATMENT_ARM, COMPARATOR_ARM)

#: Scenario codes: C0 (v1.10 D1 clean reference, erratum-A E1) + S0-S4.
SCENARIOS = ("C0", "S0", "S1", "S2", "S3", "S4")
REFERENCE_SCENARIO = "C0"
S0_SCENARIO = "S0"
ATTACK_SCENARIOS = ("S1", "S2", "S3", "S4")

#: § 4.1 gate bars — inherited from the 2026-05-27 base registration,
#: unchanged through every amendment. Boundary semantics are pre-registered
#: as INCLUSIVE: median exactly 0.05 passes (>=), p exactly 0.05 passes (<=).
MEDIAN_BAR = 0.05
P_BAR = 0.05
FINAL_WINDOW = 5

#: § 6 item 4: a C0/S0 cell whose acc_final5 sits STRICTLY more than 5 pp
#: below its own peak rolling-5-round mean is flagged (non-gating).
ANOMALY_GAP = 0.05

#: § 4.3-adjacent pre-registered attributive contrasts, reported on the same
#: endpoints, NON-GATING. For a pair (X, Y) the reported delta per (scenario,
#: seed) is degradation(Y) - degradation(X): positive = X degrades less.
ATTRIBUTIVE_CONTRASTS = (
    ("1v5", "h2p_fp_krum", "h2p_krum"),    # FP's marginal under Krum
    ("6v9", "h2p_fp_ts", "h2p_ts"),        # FP's marginal under TrustScore
    ("3v1", "h2p_fp", "h2p_fp_krum"),      # does Krum help/hurt/nothing
    ("6v7", "h2p_fp_ts", "trustscore"),    # combined-treatment generality
    ("4v1", "krum_tge_fp", "h2p_fp_krum"), # legacy composition vs corrected
)


# ===========================================================================
# Redaction (H3 scorer pattern, extended to paths)
# ===========================================================================

def redacted(value: Optional[Any]) -> str:
    """A stable, non-reversible fingerprint of a run-identifying string.

    run_uids embed launch seeds in clear text, and launch tooling embeds the
    seed in unit FILENAMES, so refusals identify rows and files by SHA-256
    prefix only: enough to locate the offender offline by hashing candidates,
    never enough to reveal a sealed seed (H3 scorer `_redacted` pattern).
    """
    if value is None:
        return "<absent>"
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()[:16]


# ===========================================================================
# Manifest reads
# ===========================================================================

def split_manifest_sha256() -> str:
    """sha256 of the committed sealed-split manifest's BYTES (erratum-A E4).

    Every unit's `eval_split_manifest_sha256` must EQUAL this; a scorer run
    in a tree without the manifest cannot corroborate anything and refuses.
    """
    path = Path(SPLIT_MANIFEST_PATH)
    if not path.exists():
        raise ScoringError(
            f"committed sealed-split manifest not found at {path} — the E4 "
            "custody equality cannot be verified without it; refusing."
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_seed_list() -> List[int]:
    """The sealed confirmatory seeds, SORTED. Values are never printed;
    outputs reference seeds only as `seed_ordinal` = index in this list."""
    path = Path(SEED_MANIFEST_PATH)
    if not path.exists():
        raise ScoringError(
            f"registered seed manifest not found at {path} — the census "
            "cannot be verified without it."
        )
    try:
        import json
        values = json.loads(path.read_text())[SEED_MANIFEST_KEY]
    except Exception as exc:  # malformed JSON or missing key alike
        raise ScoringError(
            f"registered seed manifest {path} is unreadable or lacks "
            f"{SEED_MANIFEST_KEY!r}: {type(exc).__name__}"
        ) from exc
    if (not isinstance(values, list) or not values
            or not all(isinstance(v, int) and not isinstance(v, bool)
                       for v in values)):
        raise ScoringError(
            f"registered seed manifest {path} key {SEED_MANIFEST_KEY!r} is "
            "empty or non-integer — refusing to verify against a malformed "
            "manifest (values not printed)."
        )
    if len(set(values)) != len(values):
        raise ScoringError(
            f"registered seed manifest {path} key {SEED_MANIFEST_KEY!r} "
            "contains duplicate values (values not printed) — an ambiguous "
            "manifest cannot anchor the census."
        )
    return sorted(values)


# ===========================================================================
# Endpoints
# ===========================================================================

def _round_of(entry: Mapping[str, Any], unit_ref: str) -> int:
    """A VALIDATED integral round identifier — never a coercion.

    `int` would silently turn 1.9 into 1 and True into 1, corrupting
    round ordering and the duplicate-round guard.
    Accepted: int (bool excluded) or a float that `.is_integer` (a JSON
    round-trip artifact like 7.0 -> 7). REFUSED: bools, fractional floats,
    strings and everything else — the Lane-B emitter writes ints
    (`parse_eval_trajectory`), so a string round is a malformed unit, not
    a format to accommodate.
    """
    value = entry["round"]
    if isinstance(value, bool):
        pass  # falls through to the refusal below
    elif isinstance(value, int):
        return value
    elif (isinstance(value, float) and math.isfinite(value)
            and value.is_integer()):
        return int(value)
    raise ScoringError(
        f"{unit_ref}: trajectory 'round' {value!r} is not an integral "
        "round number — bools, fractional floats and strings are refused, "
        "never coerced (a silent int() here would corrupt round ordering "
        "and deduplication)."
    )


def _sorted_entries(trajectory: Sequence[Mapping[str, Any]],
                    unit_ref: str) -> List[Mapping[str, Any]]:
    """Trajectory entries sorted by round, with the identity checks every
    endpoint needs. `unit_ref` is a pre-redacted unit label for refusals."""
    if not isinstance(trajectory, (list, tuple)) or not trajectory:
        raise ScoringError(
            f"{unit_ref}: trajectory is empty or not a list — no endpoint "
            "can be computed from it."
        )
    entries = []
    for entry in trajectory:
        if not isinstance(entry, Mapping) or "round" not in entry:
            raise ScoringError(
                f"{unit_ref}: trajectory entry without a 'round' field — the "
                "final-5 window is round-ordered and cannot be located."
            )
        entries.append(entry)
    entries.sort(key=lambda e: _round_of(e, unit_ref))
    rounds = [_round_of(e, unit_ref) for e in entries]
    if len(set(rounds)) != len(rounds):
        raise ScoringError(
            f"{unit_ref}: duplicate round numbers in the trajectory — the "
            "final-5 window is ambiguous; refusing."
        )
    return entries


def _validated_metric(value: Any, key: str, entry: Mapping[str, Any],
                      unit_ref: str) -> float:
    """A finite metric within [0.0, 1.0] inclusive, or a refusal.

    NaN and inf pass a bare isinstance check and then poison a median
    SILENTLY (NaN comparisons are all false); out-of-range values are not
    accuracies at all. A malformed value must refuse,
    never adjudicate. Boundary values 0.0 and 1.0 are valid.
    """
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0):
        raise ScoringError(
            f"{unit_ref}: trajectory entry at round {entry.get('round')!r} "
            f"carries a malformed {key!r} value ({value!r}) — every scored "
            "metric must be finite and within [0.0, 1.0] inclusive; NaN or "
            "an out-of-range value would flow into the primary gate "
            "silently, so it refuses instead."
        )
    return float(value)


def _accuracy_of(entry: Mapping[str, Any], unit_ref: str) -> float:
    return _validated_metric(entry.get("accuracy"), "accuracy", entry,
                             unit_ref)


def _last5(trajectory: Sequence[Mapping[str, Any]],
           unit_ref: str) -> List[Mapping[str, Any]]:
    entries = _sorted_entries(trajectory, unit_ref)
    if len(entries) < FINAL_WINDOW:
        raise ScoringError(
            f"{unit_ref}: trajectory has fewer than {FINAL_WINDOW} scored "
            f"rounds ({len(entries)}) — acc_final5 (§ 4.1) is undefined on a "
            "short trajectory and a truncated unit is refused, never padded."
        )
    return entries[-FINAL_WINDOW:]


def acc_final5(trajectory: Sequence[Mapping[str, Any]], unit_ref: str) -> float:
    """§ 4.1 primary endpoint: mean sealed-test accuracy over the final 5
    rounds (pre-registered robustness against the final-round-dip
    class)."""
    window = _last5(trajectory, unit_ref)
    return sum(_accuracy_of(e, unit_ref) for e in window) / FINAL_WINDOW


def final_round_accuracy(trajectory: Sequence[Mapping[str, Any]],
                         unit_ref: str) -> float:
    """§ 4.1 reported variant: the single final round, for direct Szeląg
    comparability. Non-gating."""
    entries = _sorted_entries(trajectory, unit_ref)
    return _accuracy_of(entries[-1], unit_ref)


def f1_final5(trajectory: Sequence[Mapping[str, Any]],
              unit_ref: str) -> Optional[float]:
    """The F1-based § 4.1 analog (continuity with the original F1
    registration). ABSENT f1 on any final-5 entry returns None — the
    secondary is then reported null-with-reason, never fabricated. A
    PRESENT but malformed f1 (non-numeric, non-finite, outside [0, 1])
    REFUSES: a value that exists but cannot be a score is a broken unit,
    not a legacy trajectory."""
    window = _last5(trajectory, unit_ref)
    values = []
    for entry in window:
        if "f1" not in entry:
            return None
        values.append(_validated_metric(entry["f1"], "f1", entry, unit_ref))
    return sum(values) / FINAL_WINDOW


def rolling5_peak(trajectory: Sequence[Mapping[str, Any]],
                  unit_ref: str) -> float:
    """The cell's own peak rolling-5-round mean accuracy (§ 6 item 4),
    over consecutive round-ordered windows."""
    entries = _sorted_entries(trajectory, unit_ref)
    if len(entries) < FINAL_WINDOW:
        raise ScoringError(
            f"{unit_ref}: trajectory has fewer than {FINAL_WINDOW} rounds — "
            "no rolling-5 window exists."
        )
    accs = [_accuracy_of(e, unit_ref) for e in entries]
    return max(
        sum(accs[i:i + FINAL_WINDOW]) / FINAL_WINDOW
        for i in range(len(accs) - FINAL_WINDOW + 1)
    )


def is_reference_anomaly(trajectory: Sequence[Mapping[str, Any]],
                         unit_ref: str) -> Dict[str, Any]:
    """§ 6 item 4 (`reference_anomaly`): flag a C0/S0 cell whose acc_final5
    sits STRICTLY more than ANOMALY_GAP below its own rolling-5 peak — the
    sustained absorbing-state signature. Non-gating; the § 4.1
    formula is computed as registered regardless of flags."""
    final5 = acc_final5(trajectory, unit_ref)
    peak = rolling5_peak(trajectory, unit_ref)
    gap = peak - final5
    return {
        "acc_final5": final5,
        "peak_rolling5": peak,
        "gap": gap,
        "flagged": gap > ANOMALY_GAP,
    }


# ===========================================================================
# Statistics
# ===========================================================================

def exact_median(values: Sequence[float]) -> float:
    """Plain median; n=10 gives the mean of the 5th and 6th order statistics."""
    if not values:
        raise ScoringError("median of an empty sequence is undefined")
    return float(statistics.median(values))


def exact_wilcoxon_two_sided(diffs: Sequence[float]) -> Dict[str, Any]:
    """Exact paired two-sided Wilcoxon signed-rank test (no scipy).

    Conventions — chosen to MIRROR the frozen one-sided implementation in
    `reproduction/protocol/h2prime/revalidate_v115.py::
    exact_wilcoxon_onesided` exactly, then extended two-sided:

    * **Zero diffs are DROPPED before ranking** (Wilcoxon's original
      zero-exclusion — the frozen implementation's behavior; not the Pratt
      variant). `n_zero_dropped` reports how many.
    * **Ties in |d| receive mid-ranks** (average of the run of tied
      positions), identical to the frozen implementation's grouping.
    * **W+** = sum of the (mid-)ranks of the strictly positive diffs.
    * **Null distribution**: exact enumeration of all 2^n equiprobable sign
      assignments over the REALIZED rank vector (ties included), the same
      subset enumeration the frozen implementation uses. Both tail
      probabilities INCLUDE the observed statistic (>= / <=), matching the
      frozen implementation's `>=` upper tail.
    * **Two-sided p** = min(1, 2 * min(P(W+ <= w_obs), P(W+ >= w_obs))) —
      the standard exact doubling convention; the mid-rank null is symmetric
      so doubling the smaller tail is well-defined.
    * **Degenerate n = 0** (all diffs zero, or no pairs): W+ = 0, p = 1.0 —
      no evidence in either direction, matching the frozen implementation's
      (0.0, 1.0).

    n here is at most 10 (one pair per sealed seed), so full enumeration
    (2^10 = 1024 subsets) is exact and instant.
    """
    nz = [float(d) for d in diffs if float(d) != 0.0]
    n = len(nz)
    result: Dict[str, Any] = {
        "method": (
            "exact paired two-sided Wilcoxon signed-rank; zeros dropped, "
            "mid-ranks for |d| ties, full 2^n sign enumeration, "
            "p = min(1, 2*min(lower_tail, upper_tail)), tails inclusive"
        ),
        "n_pairs": len(list(diffs)),
        "n_zero_dropped": len(list(diffs)) - n,
        "n_effective": n,
    }
    if n == 0:
        result.update({"w_plus": 0.0, "p_ge": 1.0, "p_le": 1.0,
                       "p_two_sided": 1.0})
        return result

    # Mid-ranks over |d| — the frozen implementation's grouping, verbatim.
    ranks: Dict[int, float] = {}
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    rank_values = sorted(ranks.values())
    count_ge = 0
    count_le = 0
    total = 0
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            total += 1
            s = sum(rank_values[c] for c in combo)
            if s >= w_plus:
                count_ge += 1
            if s <= w_plus:
                count_le += 1
    p_ge = count_ge / total
    p_le = count_le / total
    result.update({
        "w_plus": float(w_plus),
        "p_ge": p_ge,
        "p_le": p_le,
        "p_two_sided": min(1.0, 2.0 * min(p_ge, p_le)),
    })
    return result


def scenario_gate(median_value: float, p_value: float) -> Dict[str, Any]:
    """§ 4.1 per-scenario gate, boundary-inclusive as registered:
    median >= 0.05 (5 pp) AND exact two-sided Wilcoxon p <= 0.05."""
    median_pass = median_value >= MEDIAN_BAR
    p_pass = p_value <= P_BAR
    return {
        "median_bar": MEDIAN_BAR,
        "p_bar": P_BAR,
        "median_pass": median_pass,
        "p_pass": p_pass,
        "passed": median_pass and p_pass,
    }
