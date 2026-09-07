"""Per-defense run-to-run variance-envelope reduction.

Quantifies the run-to-run variance envelope of the FL pipeline per defense, to
GATE the H2 dev-sweep sizing decision (how many replicates the confirmatory
matrix needs to resolve a real effect above pipeline noise). The study is
scenario S4 only (the registered `--scenario`, default ``S4_full_mix``, is
enforced on BOTH inputs of every unit as a chain-of-custody gate), over the
four defenses {krum, trustscore, tge, krum_tge}, across two sibling EXP
matrices reduced JOINTLY here:

    Arm A : seed 42 x R repeats per defense -> pure run-to-run noise sigma_run
    Arm B : 5 dev seeds x 1 each per defense -> sigma_total = sigma_seed  run
    Decomposition: sigma_seed^2 = sigma_total^2 - sigma_run^2.

This script is FIXTURE-DRIVEN: it consumes a manifest of local per-unit result
JSONs + signal-log JSONLs, so it can be exercised entirely from synthetic
fixtures before the real AWS study exists.

METRICS
-------
PRIMARY recall_coldstart : recall@10%FPR of malicious-client detection, in
                              the cold-start scope (tenure in [1, k], v1.3 F6),
                              scored against a FIXED PROVISIONAL per-defense
                              threshold supplied via --thresholds. The cut is
                              applied UNCHANGED to every replicate. It is NEVER
                              derived from the analysed logs and NEVER
                              re-centred per replicate — doing so would
                              normalise away exactly the threshold-crossing
                              variance this study measures. (This module
                              therefore never imports the dev-gate ramp-selection
                              analysis nor any threshold-SELECTION helper — a
                              guard test pins the literal absence of those
                              module names from this source.)
secondary recall_allrounds : same metric over all rounds.
x-check auc_coldstart / : threshold-FREE per-replicate AUC of the trust
          auc_allrounds score vs malicious ground truth (and its SD),
                               so the reported SD is shown NOT to be an artifact
                               of the chosen cut.
secondary final_accuracy / : finals from the result JSON (continuity with
          final_f1 EXP-005c/e).

SCORE DIRECTION (the Szelag-traceback landmine — verified against the emitting
code, not assumed). Every praxis defense exposes a TRUST score (HIGHER = kept);
a malicious client is DETECTED when its score is BELOW the cut
(flag = score < threshold):

    defense score field direction source (flowerfl/)
    krum krum_score trust byzantine_defense.py:303-316
    trustscore trust_score trust byzantine_defense.py:464-481
    tge tge_score trust byzantine_defense.py:533; scenario_strategy.py:636-639
    krum_tge tge_score trust composed chain final decision (byzantine_defense.py:842-893
                                        sequential; TGE last plugin). Krum-filtered rows carry a
                                        null tge_score and drop out of the population (the
                                        is_tge_scored null-passthrough convention; ramp-selection
                                        prior art scripts/analyze_ramp_*.py:72-79).

`effective_weight` reflects data counts, NOT a keep/exclude flag, and is never
used for detection. Cold-start scope keys on the signal log's `tenure` field
(rounds since this LOGICAL identity (re)joined, restarting at 1 on a rejoin —
flowerfl/scenario_strategy.py:67-73), which is the logged encoding of
per-round client presence; `server_round = scenario_round + 1` in Flower mode
(discovery-round offset) is therefore never recomputed here — `tenure` is read
directly, matching scripts/compute_recall_fpr.py:64-68.

Null-score rows are excluded from a defense's evaluation population, matching
scripts/compute_recall_fpr.py:78-80.

Usage:
    python scripts/analyze_variance_envelope.py \\
        --manifest variance_manifest.json \\
        --thresholds provisional_thresholds.json \\
        --out results/<date>/variance_envelope/envelope.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

# --- pre-registered constants (do not edit without an amendment) ------------
DEFAULT_SCENARIO: str = "S4_full_mix"  # registered S4 scenario (this study is S4-only)
DEFAULT_COLDSTART_K: int = 3          # v1.3 F6 cold-start window
DEFAULT_TARGET_FPR: float = 0.10      # the FPR the provisional threshold targets
DEFAULT_BOOTSTRAP_N: int = 2000       # bootstrap resamples for the CI of the mean
DEFAULT_BOOTSTRAP_SEED: int = 12345   # seeded RNG for test determinism
CI_ALPHA: float = 0.05                # 95% CI

PRIMARY_METRIC: str = "recall_coldstart"
METRICS: tuple[str, ...] = (
    "recall_coldstart",
    "recall_allrounds",
    "auc_coldstart",
    "auc_allrounds",
    "final_accuracy",
    "final_f1",
)
ARMS: tuple[str, ...] = ("A", "B")

_SIGMA_SEED_ZERO_NOTE = (
    "sigma_seed ~ 0 (seed effect indistinguishable from run noise; "
    "sigma_total^2 - sigma_run^2 <= 0)"
)


class VarianceEnvelopeError(RuntimeError):
    """Raised when the study inputs violate a pre-registration precondition."""


# ---------------------------------------------------------------------------
# frozen value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefenseScoreSpec:
    """Per-defense score field + detection direction + artifact-identity
    vocabulary, each citing the emitting source line so it is auditable
    (Szelag-traceback discipline).

    `result_config` and `row_defense_token` pin the EXACT (no fuzzy
    normalization) identity tokens a genuine artifact of this defense carries,
    so a swapped-defense artifact (e.g. tge logs under a krum_tge unit — both
    carry tge_score rows, so the score field alone cannot discriminate) is
    rejected. Crucially the signal-row token is the STRATEGY CLASS name lowered
    (server_app.py:78: strategy_name.replace("Scenario","").lower), NOT the
    config label — so tge -> "tgensemble" and krum_tge -> "krumtge", not the
    naive config tokens."""
    defense: str
    score_field: str
    higher_is_trust: bool  # True => flag (detect) when score < threshold
    source: str
    result_config: str       # result JSON "config" label (run_phase4_flower.py:1217)
    row_defense_token: str   # signal-row "defense" token (server_app.py:78)
    identity_source: str


# The one table-like constant that pins score field + direction + identity
# vocabulary per defense. Identity evidence: SUPPORTED_CONFIGS
# run_phase4_flower.py:210-218; build_strategy_for_config class mapping :29-78
# (ScenarioKrum/ScenarioTrustScore/ScenarioTGEnsemble/ScenarioKrumTGE);
# server_app.py:78 row-token derivation; result "config" written at :1217.
DEFENSE_SCORE_SPECS: dict[str, DefenseScoreSpec] = {
    "krum": DefenseScoreSpec(
        "krum", "krum_score", True,
        "flowerfl/byzantine_defense.py:303-316 (1-normalised sum-of-distances; higher=consensus)",
        result_config="Krum", row_defense_token="krum",
        identity_source="config run_phase4_flower.py:210-218; token ScenarioKrum->server_app.py:78"),
    "trustscore": DefenseScoreSpec(
        "trustscore", "trust_score", True,
        "flowerfl/byzantine_defense.py:464-481 (EMA reputation; 1.0=close-to-median)",
        result_config="TrustScore", row_defense_token="trustscore",
        identity_source="config run_phase4_flower.py:210-218; token ScenarioTrustScore->server_app.py:78"),
    "tge": DefenseScoreSpec(
        "tge", "tge_score", True,
        "flowerfl/byzantine_defense.py:533 + scenario_strategy.py:636-639 (final_score; decision=score>=thr)",
        result_config="TGE", row_defense_token="tgensemble",  # NOT "tge"
        identity_source="config run_phase4_flower.py:210-218; token ScenarioTGEnsemble->server_app.py:78"),
    "krum_tge": DefenseScoreSpec(
        "krum_tge", "tge_score", True,
        "composed chain final decision = tge_score (byzantine_defense.py:842-893 sequential; TGE last)",
        result_config="Krum+TGE", row_defense_token="krumtge",  # NOT "krum_tge"
        identity_source="config run_phase4_flower.py:210-218; token ScenarioKrumTGE->server_app.py:78"),
}

# Reverse of the R6 identity table: result-config LABEL -> defense id. Used to
# canonicalize threshold/anchor input keys, because the frozen-threshold
# pipeline keys those files by config label ("Krum"/"TGE"/...) rather than the
# defense id ("krum"/"tge"/...). All four result_config labels are distinct, so
# this reverse map is unambiguous.
_CONFIG_LABEL_TO_DEFENSE: dict[str, str] = {
    s.result_config: d for d, s in DEFENSE_SCORE_SPECS.items()
}


@dataclass(frozen=True)
class UnitSpec:
    arm: str
    defense: str
    seed: int
    replicate: int
    result_path: Path
    signal_path: Path

    @property
    def label(self) -> str:
        return f"{self.arm}/{self.defense}/seed{self.seed}/rep{self.replicate}"


@dataclass(frozen=True)
class RecallResult:
    recall: float
    fpr: Optional[float]  # None when no honest rows in scope (never NaN -> strict JSON)
    n_malicious: int
    n_honest: int
    n_flagged_malicious: int


@dataclass(frozen=True)
class Replicate:
    arm: str
    defense: str
    seed: int
    replicate: int
    metrics: dict[str, Optional[float]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeanCI:
    mean: Optional[float]
    sd: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "sd": self.sd,
                "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}


@dataclass(frozen=True)
class Decomposition:
    tier: str
    sigma_run: Optional[float]
    sigma_total: Optional[float]
    sigma_seed: Optional[float]
    sigma_seed_var: Optional[float]
    note: Optional[str]
    sigma_run_source: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "sigma_run": self.sigma_run,
            "sigma_total": self.sigma_total,
            "sigma_seed": self.sigma_seed,
            "sigma_seed_var": self.sigma_seed_var,
            "note": self.note,
            "sigma_run_source": self.sigma_run_source,
        }


# ---------------------------------------------------------------------------
# signal-log row helpers
# ---------------------------------------------------------------------------

def tenure_of(row: dict) -> Optional[int]:
    """Rounds since this logical identity (re)joined (1-indexed). Prefers the
    v3 `tenure` field; falls back to the legacy `rounds_since_join`."""
    val = row.get("tenure")
    if val is None:
        val = row.get("rounds_since_join")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def in_coldstart(row: dict, k: int) -> bool:
    """Cold-start scope (v1.3 F6): tenure in [1, k]. A rejoin restarts tenure
    at 1, so a client rejoining at round r contributes rounds r..r+k-1."""
    t = tenure_of(row)
    return t is not None and 1 <= t <= k


def score_of(row: dict, field_name: str) -> Optional[float]:
    """Numeric detection score for `field_name`, or None if absent/null."""
    val = row.get(field_name)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _scored_rows(rows: Iterable[dict], spec: DefenseScoreSpec, *,
                 coldstart_only: bool, k: int) -> list[tuple[float, bool]]:
    """(score, is_malicious) for rows in scope with a non-null score."""
    out: list[tuple[float, bool]] = []
    for row in rows:
        if coldstart_only and not in_coldstart(row, k):
            continue
        s = score_of(row, spec.score_field)
        if s is None:  # null score => not in this detector's population
            continue
        out.append((s, bool(row.get("malicious_gt"))))
    return out


def _is_flagged(score: float, threshold: float, spec: DefenseScoreSpec) -> bool:
    """Detection decision. TRUST scores flag (detect) BELOW the cut; a
    hypothetical distance-style score would flag ABOVE it."""
    if spec.higher_is_trust:
        return score < threshold
    return score > threshold


def recall_at_fixed_threshold(rows: Sequence[dict], spec: DefenseScoreSpec,
                              threshold: float, *, coldstart_only: bool,
                              k: int) -> Optional[RecallResult]:
    """Recall + FPR of malicious detection at a FIXED threshold (never
    re-derived). Returns None when there are no malicious rows in scope
    (undefined recall) — graceful, never a crash."""
    scored = _scored_rows(rows, spec, coldstart_only=coldstart_only, k=k)
    mal = [s for s, m in scored if m]
    honest = [s for s, m in scored if not m]
    if not mal:
        return None
    tp = sum(1 for s in mal if _is_flagged(s, threshold, spec))
    fp = sum(1 for s in honest if _is_flagged(s, threshold, spec))
    # None (not NaN) when no honest rows in scope, so the report stays strict
    # JSON (json.dumps(..., allow_nan=False) round-trips for downstream parsers)
    fpr = fp / len(honest) if honest else None
    return RecallResult(recall=tp / len(mal), fpr=fpr,
                        n_malicious=len(mal), n_honest=len(honest),
                        n_flagged_malicious=tp)


def detector_auc(rows: Sequence[dict], spec: DefenseScoreSpec, *,
                 coldstart_only: bool, k: int) -> Optional[float]:
    """Threshold-free detector AUC, direction-aware. Equals the probability
    that a random malicious client is scored MORE anomalous than a random
    honest one (ties = 0.5). For TRUST scores "more anomalous" = LOWER trust,
    so a perfect detector -> 1.0, random -> 0.5. Returns None if either class
    is empty in scope."""
    scored = _scored_rows(rows, spec, coldstart_only=coldstart_only, k=k)
    mal = [s for s, m in scored if m]
    honest = [s for s, m in scored if not m]
    if not mal or not honest:
        return None
    wins = 0.0
    for sm in mal:
        for sh in honest:
            more_anomalous = (sm < sh) if spec.higher_is_trust else (sm > sh)
            tie = sm == sh
            if more_anomalous:
                wins += 1.0
            elif tie:
                wins += 0.5
    return wins / (len(mal) * len(honest))


def _finite_or_none(val: Any) -> Optional[float]:
    """Coerce to a finite float, else None. Non-finite (NaN/inf) finals from a
    degenerate run are treated as missing — mirrors flowerfl/signal_logger.py
    _jsonify and keeps the report strict JSON."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def finals_from_result(result: dict) -> tuple[Optional[float], Optional[float]]:
    """(final_accuracy, final_f1) from a result JSON; None if absent or
    non-finite."""
    acc = _finite_or_none(result.get("final_accuracy", result.get("final_acc")))
    f1 = _finite_or_none(result.get("final_f1"))
    return acc, f1


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_ci_mean(values: Sequence[float], *, n: int, rng_seed: int,
                      alpha: float = CI_ALPHA) -> tuple[float, float]:
    """Seeded, percentile bootstrap CI of the MEAN. Deterministic for a given
    (values, n, rng_seed). Degenerates to (v, v) for zero-variance input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci_mean requires at least one value")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def mean_sd_ci(values: Sequence[float], *, bootstrap_n: int,
               rng_seed: int) -> MeanCI:
    """Mean, sample SD (ddof=1), bootstrap 95% CI of the mean, and n. SD is
    None for n<2 (ddof=1 undefined); all fields None for n=0."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return MeanCI(None, None, None, None, 0)
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n >= 2 else None
    lo, hi = bootstrap_ci_mean(vals, n=bootstrap_n, rng_seed=rng_seed)
    return MeanCI(mean=mean, sd=sd, ci_low=lo, ci_high=hi, n=n)


def decompose(sigma_run: Optional[float], sigma_total: Optional[float], *,
              tier: str, sigma_run_source: Optional[str] = None) -> Decomposition:
    """Assemble a variance decomposition. sigma_seed^2 = sigma_total^2 -
    sigma_run^2 whenever BOTH sigmas are available; when the difference is
    non-positive, sigma_seed is reported as None with an HONEST note and the
    raw (possibly negative) variance surfaced in sigma_seed_var — never
    clamped silently, never crashed."""
    sigma_seed: Optional[float] = None
    sigma_seed_var: Optional[float] = None
    note: Optional[str] = None

    if sigma_run is not None and sigma_total is not None:
        sigma_seed_var = sigma_total ** 2 - sigma_run ** 2
        if sigma_seed_var > 0:
            sigma_seed = float(np.sqrt(sigma_seed_var))
        else:
            note = _SIGMA_SEED_ZERO_NOTE  # honest: do not clamp to a fake value
    elif sigma_run is not None and sigma_total is None:
        note = "sigma_total / sigma_seed unavailable (Arm B absent)"
    elif sigma_total is not None and sigma_run is None:
        note = ("sigma_run unavailable (Arm A absent; supply this defense's "
                "--sigma-run-anchors entry from the EXP-005c/005e pair to "
                "resolve sigma_seed)")
    else:
        note = "insufficient replicates to estimate any variance component"

    return Decomposition(tier=tier, sigma_run=sigma_run, sigma_total=sigma_total,
                         sigma_seed=sigma_seed, sigma_seed_var=sigma_seed_var,
                         note=note, sigma_run_source=sigma_run_source)


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------

def _resolve(base: Path, ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else (base / p)


def _coerce_float_mapping(mapping: dict, what: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in mapping.items():
        try:
            f = float(val)
        except (TypeError, ValueError):
            raise VarianceEnvelopeError(
                f"{what} for {key!r} is not a number: {val!r}")
        # json.loads accepts -Infinity/Infinity/NaN by default; the frozen
        # threshold artifact emits -inf when its cut is frozen from an empty
        # honest-score population. A -inf cut silently reports zero recall on
        # every replicate (score < -inf is always false), so reject at load
        # rather than let it corrupt the reduction and only surface at the
        # allow_nan=False --out write.
        if not math.isfinite(f):
            raise VarianceEnvelopeError(
                f"{what} for {key!r} is not finite: {f} — a non-finite cut/anchor "
                f"would silently corrupt the reduction (a -inf cut reports zero "
                f"recall on every replicate); the artifact is malformed, likely "
                f"frozen from an empty honest-score population")
        out[str(key)] = f
    return out


def _canonical_defense_key(key: str) -> str:
    """Map an input key to a defense id: an id passes through; an EXACT
    result-config label ("Krum"/"TGE"/...) maps to its defense id (no fuzzy
    matching, per the R6 identity stance); anything else is left unchanged."""
    if key in DEFENSE_SCORE_SPECS:
        return key
    if key in _CONFIG_LABEL_TO_DEFENSE:
        return _CONFIG_LABEL_TO_DEFENSE[key]
    return key


def _canonicalize_defense_keys(mapping: dict[str, float], what: str) -> dict[str, float]:
    """Canonicalize config-label keys to defense ids. Two source keys landing on
    the same defense id with DISAGREEING values -> VarianceEnvelopeError naming
    both originals; equal values collapse to one. Unknown keys pass through
    unchanged (thresholds: harmlessly ignored by the membership check; anchors:
    still caught by analyze's fail-fast on unknown defenses)."""
    out: dict[str, float] = {}
    origin: dict[str, str] = {}  # canonical id -> original key that set it
    for key, val in mapping.items():
        canon = _canonical_defense_key(key)
        if canon in out and out[canon] != val:
            raise VarianceEnvelopeError(
                f"{what} keys {origin[canon]!r} and {key!r} both map to defense "
                f"{canon!r} with disagreeing values {out[canon]} vs {val}")
        if canon not in out:
            out[canon] = val
            origin[canon] = key
    return out


def load_thresholds(path: Path) -> dict[str, float]:
    """Load the FIXED provisional per-defense cuts. Accepts BOTH shapes:

    - the simple mapping ``{defense: cut}``; and
    - the frozen threshold artifact wrapper ``{"_meta": {...}, "thresholds":
      {defense: cut}}`` (the artifact the cut is frozen into once, from the
      EXP-005c/005e population, per spec § 5) — top-level keys starting with
      ``_`` are ignored, and any OTHER non-underscore top-level key alongside
      ``thresholds`` is rejected as ambiguous.

    Reading the fixed input's SHAPE is not threshold DERIVATION: the cut is
    still supplied, never computed from the analysed logs.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise VarianceEnvelopeError(
            f"--thresholds must be a JSON object mapping defense -> cut, got {type(raw).__name__}")
    if "thresholds" in raw:
        inner = raw["thresholds"]
        if not isinstance(inner, dict):
            raise VarianceEnvelopeError(
                "frozen threshold artifact 'thresholds' must be an object "
                f"mapping defense -> cut, got {type(inner).__name__}")
        stray = [k for k in raw if k != "thresholds" and not str(k).startswith("_")]
        if stray:
            raise VarianceEnvelopeError(
                f"ambiguous threshold artifact: unexpected top-level key(s) {stray} "
                f"alongside 'thresholds' (only underscore-prefixed metadata may coexist)")
        mapping = inner
    else:
        mapping = raw
    return _canonicalize_defense_keys(
        _coerce_float_mapping(mapping, "threshold"), "threshold")


def load_sigma_run_anchors(path: Path) -> dict[str, float]:
    """Load the per-defense external sigma_run anchors (EXP-005c/005e same-seed
    pair). A plain ``{defense: sigma_run}`` mapping — the anchor is a PER-DEFENSE
    quantity (it spans ~0.008 for trustscore to ~0.157 for krum_tge), so a
    single scalar shared across defenses would corrupt each defense's
    sigma_seed = sqrt(sigma_total^2 - sigma_run^2)."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise VarianceEnvelopeError(
            f"--sigma-run-anchors must be a JSON object mapping defense -> "
            f"sigma_run, got {type(raw).__name__}")
    return _canonicalize_defense_keys(
        _coerce_float_mapping(raw, "sigma_run anchor"), "sigma_run anchor")


def _replicate_ordinal(u: dict, i: int) -> int:
    """Resolve the replicate ordinal, accepting BOTH this module's `replicate`
    key and the launch-side `repeat` key (praxis_exp.units.Unit serialised via
    asdict names the field `repeat`). If both are present and DISAGREE, fail
    fast naming both values; if either alone is present use it; absent -> 0."""
    has_rep = "replicate" in u
    has_repeat = "repeat" in u
    if has_rep and has_repeat:
        rep = int(u["replicate"])
        rpt = int(u["repeat"])
        if rep != rpt:
            raise VarianceEnvelopeError(
                f"manifest unit #{i}: conflicting replicate ordinal — "
                f"'replicate'={rep} vs 'repeat'={rpt}; the two keys must agree")
        return rep
    if has_rep:
        return int(u["replicate"])
    if has_repeat:
        return int(u["repeat"])
    return 0


def _unit_defense(u: dict, i: int) -> str:
    """Resolve the canonical defense id, accepting BOTH this module's `defense`
    key and the launch-side `config` key (praxis_exp.units.Unit serialises the
    defense under `config`). Each is canonicalized (config labels -> ids, R7); if
    both are present and DISAGREE after canonicalization, fail fast naming both;
    absent from both -> malformed-unit error naming both keys."""
    has_def = "defense" in u
    has_cfg = "config" in u
    if has_def and has_cfg:
        d = _canonical_defense_key(str(u["defense"]))
        c = _canonical_defense_key(str(u["config"]))
        if d != c:
            raise VarianceEnvelopeError(
                f"manifest unit #{i}: conflicting defense — 'defense'={u['defense']!r} "
                f"(->{d!r}) vs 'config'={u['config']!r} (->{c!r}); the two keys must agree")
        return d
    if has_def:
        return _canonical_defense_key(str(u["defense"]))
    if has_cfg:
        return _canonical_defense_key(str(u["config"]))
    raise VarianceEnvelopeError(
        f"manifest unit #{i} is malformed: missing the defense (neither "
        f"'defense' nor the launch-side 'config' key is present)")


def load_manifest(path: Path) -> list[UnitSpec]:
    path = Path(path)
    base = path.parent
    raw = json.loads(path.read_text())
    units_raw = raw.get("units") if isinstance(raw, dict) else raw
    if not isinstance(units_raw, list) or not units_raw:
        raise VarianceEnvelopeError(
            "manifest must contain a non-empty 'units' list")
    units: list[UnitSpec] = []
    for i, u in enumerate(units_raw):
        try:
            arm = str(u["arm"]).upper()
            # canonicalize config-label defense keys (launch-side matrix docs
            # list defenses as "Krum"/"TGE"/... ) to defense ids, reusing the R7
            # helper; accepts the launch-side `config` key; unknown values pass
            # through to analyze's fail-fast
            defense = _unit_defense(u, i)
            seed = int(u["seed"])
            replicate = _replicate_ordinal(u, i)
            result_ref = str(u["result"])
            signal_ref = str(u["signal"])
        except (KeyError, TypeError, ValueError) as e:
            raise VarianceEnvelopeError(
                f"manifest unit #{i} is malformed ({e}); required keys: "
                f"arm, (defense|config), seed, result, signal")
        if arm not in ARMS:
            raise VarianceEnvelopeError(
                f"manifest unit #{i}: arm {arm!r} not in {ARMS}")
        units.append(UnitSpec(
            arm=arm, defense=defense, seed=seed, replicate=replicate,
            result_path=_resolve(base, result_ref),
            signal_path=_resolve(base, signal_ref)))
    return units


def read_signal_rows(path: Path) -> list[dict]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# registered-scenario integrity (chain of custody on both inputs)
# ---------------------------------------------------------------------------

def _scenario_stem(value: Any) -> str:
    """Normalise a scenario token to its stem: strip any directory path and a
    trailing suffix (the result JSON records the full scenario path while
    signal rows carry the bare stem token)."""
    return Path(str(value)).stem


def _result_scenario(result: dict) -> Optional[Any]:
    """Resolve the result-side scenario. The REAL runner records it under
    provenance.scenario_path (scripts/run_phase4_flower.py:1176-1180), not a
    top-level key; a top-level `scenario` is also honoured (and preferred) for
    forward-compat. Returns None if neither is present (or provenance is not a
    dict)."""
    top = result.get("scenario")
    if top is not None:
        return top
    prov = result.get("provenance")
    if isinstance(prov, dict):
        return prov.get("scenario_path")
    return None


def _check_scenario_integrity(unit: UnitSpec, rows: Sequence[dict],
                              result: dict, expected_stem: str) -> None:
    """Reject any unit whose result JSON or signal rows are not the registered
    scenario. Both inputs carry `scenario` (result: provenance.scenario_path or
    a top-level `scenario`; rows: required field per the signal logger). A
    wrong, mixed, or MISSING scenario -> VarianceEnvelopeError naming the unit
    and offending value; a missing field is a malformed/legacy input, never
    silently assumed to match."""
    rsc = _result_scenario(result)
    if rsc is None:
        raise VarianceEnvelopeError(
            f"unit {unit.label}: result JSON {unit.result_path} has no scenario "
            f"at top-level 'scenario' nor 'provenance.scenario_path' — a missing "
            f"scenario is a malformed/legacy result; refusing to assume it is "
            f"the registered {expected_stem!r}")
    if _scenario_stem(rsc) != expected_stem:
        raise VarianceEnvelopeError(
            f"unit {unit.label}: result scenario {rsc!r} (stem {_scenario_stem(rsc)!r}) "
            f"!= registered {expected_stem!r} — wrong-scenario input")
    for j, row in enumerate(rows):
        rowsc = row.get("scenario")
        if rowsc is None:
            raise VarianceEnvelopeError(
                f"unit {unit.label}: signal row {j} has no 'scenario' (required "
                f"field per the signal logger); refusing to assume {expected_stem!r}")
        if _scenario_stem(rowsc) != expected_stem:
            raise VarianceEnvelopeError(
                f"unit {unit.label}: signal row {j} scenario {rowsc!r} (stem "
                f"{_scenario_stem(rowsc)!r}) != registered {expected_stem!r} — "
                f"wrong/mixed-scenario log")


def _seed_check(where: str, value: Any, unit: UnitSpec) -> None:
    """One seed comparison against the manifest unit's declared seed. Missing
    (a required field per the signal logger / runner) or non-integer or
    mismatched -> VarianceEnvelopeError naming the unit, the location, and both
    values."""
    if value is None:
        raise VarianceEnvelopeError(
            f"unit {unit.label}: {where} has no 'seed' (required field); refusing "
            f"to assume it is the declared seed {unit.seed}")
    try:
        got = int(value)
    except (TypeError, ValueError):
        raise VarianceEnvelopeError(
            f"unit {unit.label}: {where} seed {value!r} is not an integer")
    if got != unit.seed:
        raise VarianceEnvelopeError(
            f"unit {unit.label}: {where} seed {got} != declared manifest seed "
            f"{unit.seed} — wrong-seed artifact (would smuggle seed variance into "
            f"sigma_run)")


def _check_seed_integrity(unit: UnitSpec, rows: Sequence[dict],
                          result: dict) -> None:
    """Reject any unit whose result seed or any signal-row seed disagrees with
    the manifest's declared seed. Both inputs carry `seed` (result: top-level,
    scripts/run_phase4_flower.py:1216-1228; rows: required field per the signal
    logger, flowerfl/signal_logger.py:89)."""
    _seed_check("result JSON", result.get("seed"), unit)
    for j, row in enumerate(rows):
        _seed_check(f"signal row {j}", row.get("seed"), unit)


def _check_defense_integrity(unit: UnitSpec, rows: Sequence[dict],
                             result: dict, spec: DefenseScoreSpec) -> None:
    """Reject a swapped-defense artifact. The result JSON's `config` must EXACTLY
    equal the spec's result_config and every signal row's `defense` token must
    EXACTLY equal the spec's row_defense_token (strict equality — any future
    vocabulary drift fails loud rather than silently mislabelling a defense; the
    map is the single constant to update). A tge artifact under a krum_tge unit
    is caught here even though both carry tge_score rows."""
    cfg = result.get("config")
    if cfg is None:
        raise VarianceEnvelopeError(
            f"unit {unit.label}: result JSON has no 'config' label (required to "
            f"verify defense identity); refusing to assume {spec.result_config!r}")
    if cfg != spec.result_config:
        raise VarianceEnvelopeError(
            f"unit {unit.label}: result config {cfg!r} != expected "
            f"{spec.result_config!r} for defense {unit.defense!r} — wrong-defense "
            f"artifact (swapped-defense inputs corrupt the per-defense envelope)")
    for j, row in enumerate(rows):
        tok = row.get("defense")
        if tok is None:
            raise VarianceEnvelopeError(
                f"unit {unit.label}: signal row {j} has no 'defense' token "
                f"(required field); refusing to assume {spec.row_defense_token!r}")
        if tok != spec.row_defense_token:
            raise VarianceEnvelopeError(
                f"unit {unit.label}: signal row {j} defense token {tok!r} != "
                f"expected {spec.row_defense_token!r} for defense {unit.defense!r} "
                f"— wrong-defense log")


# ---------------------------------------------------------------------------
# per-unit reduction
# ---------------------------------------------------------------------------

def reduce_unit(unit: UnitSpec, spec: DefenseScoreSpec, threshold: float, *,
                k: int, expected_scenario_stem: str) -> Replicate:
    rows = read_signal_rows(unit.signal_path)
    result = json.loads(Path(unit.result_path).read_text())

    # chain-of-custody gate: both inputs must be the registered scenario, carry
    # the manifest unit's declared seed (wrong-seed artifacts would smuggle seed
    # variance into sigma_run), AND carry this defense's identity vocabulary (a
    # swapped-defense artifact corrupts the per-defense envelope)
    _check_scenario_integrity(unit, rows, result, expected_scenario_stem)
    _check_seed_integrity(unit, rows, result)
    _check_defense_integrity(unit, rows, result, spec)

    rc = recall_at_fixed_threshold(rows, spec, threshold, coldstart_only=True, k=k)
    ra = recall_at_fixed_threshold(rows, spec, threshold, coldstart_only=False, k=k)
    auc_cs = detector_auc(rows, spec, coldstart_only=True, k=k)
    auc_all = detector_auc(rows, spec, coldstart_only=False, k=k)
    acc, f1 = finals_from_result(result)

    metrics: dict[str, Optional[float]] = {
        "recall_coldstart": rc.recall if rc else None,
        "recall_allrounds": ra.recall if ra else None,
        "auc_coldstart": auc_cs,
        "auc_allrounds": auc_all,
        "final_accuracy": acc,
        "final_f1": f1,
    }
    diagnostics = {
        "fpr_coldstart": rc.fpr if rc else None,
        "fpr_allrounds": ra.fpr if ra else None,
        "n_malicious_coldstart": rc.n_malicious if rc else 0,
        "n_honest_coldstart": rc.n_honest if rc else 0,
        "n_malicious_allrounds": ra.n_malicious if ra else 0,
        "score_field": spec.score_field,
        "threshold": threshold,
    }
    return Replicate(arm=unit.arm, defense=unit.defense, seed=unit.seed,
                     replicate=unit.replicate, metrics=metrics,
                     diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# joint reduction of both arms
# ---------------------------------------------------------------------------

def _validate_units(units: Sequence[UnitSpec]) -> None:
    """Fail fast on manifests that violate the study's arm definitions
    (pre-registration § 9). All raise VarianceEnvelopeError so a hand-authored
    manifest never silently biases a variance component.

    1. No duplicate (arm, defense, seed, replicate) — a copy-pasted replicate
       would double-count and bias sigma.
    2. No reused result/signal artifact path across units — every unit is a
       distinct run, so a shared artifact double-counts one run as independent
       run-to-run noise and biases sigma_run downward (the runner's in-container
       filenames are seed-only, so the copy-paste is easy to make).
    3. Arm A per defense = fixed-seed repeats: all seeds must be EQUAL, else
       seed variance is smuggled into the "pure sigma_run" estimate.
    4. Arm B per defense = distinct-seed singles: seeds must be pairwise
       DISTINCT, else a mislabeled repeat inflates sigma_total.
    """
    seen: set[tuple[str, str, int, int]] = set()
    seen_result: dict[Path, str] = {}  # resolved path -> unit label
    seen_signal: dict[Path, str] = {}
    for u in units:
        key = (u.arm, u.defense, u.seed, u.replicate)
        if key in seen:
            raise VarianceEnvelopeError(
                f"duplicate unit (arm, defense, seed, replicate)={key} in manifest "
                f"— a copy-pasted replicate would double-count and bias sigma; "
                f"each unit must be unique")
        seen.add(key)
        # per-kind path-reuse check (a unit's result and signal are different
        # files, so they are tracked separately) — resolved so two different
        # relative refs to the same file still collide
        rp = u.result_path.resolve()
        if rp in seen_result:
            raise VarianceEnvelopeError(
                f"units {seen_result[rp]} and {u.label} share result artifact {rp} "
                f"— every unit is a distinct run; a reused artifact double-counts "
                f"one run and biases sigma_run")
        seen_result[rp] = u.label
        sp = u.signal_path.resolve()
        if sp in seen_signal:
            raise VarianceEnvelopeError(
                f"units {seen_signal[sp]} and {u.label} share signal artifact {sp} "
                f"— every unit is a distinct run; a reused artifact double-counts "
                f"one run and biases sigma_run")
        seen_signal[sp] = u.label

    for defense in sorted({u.defense for u in units}):
        arm_a_seeds = [u.seed for u in units if u.defense == defense and u.arm == "A"]
        arm_b_seeds = [u.seed for u in units if u.defense == defense and u.arm == "B"]
        if len(set(arm_a_seeds)) > 1:
            raise VarianceEnvelopeError(
                f"Arm A for defense {defense!r} mixes seeds {sorted(set(arm_a_seeds))} "
                f"— Arm A is DEFINED as fixed-seed repeats (pre-registration § 9); "
                f"mixed seeds would smuggle seed variance into sigma_run")
        if len(arm_b_seeds) != len(set(arm_b_seeds)):
            dupes = sorted({s for s in arm_b_seeds if arm_b_seeds.count(s) > 1})
            raise VarianceEnvelopeError(
                f"Arm B for defense {defense!r} repeats seed(s) {dupes} — Arm B is "
                f"DEFINED as distinct-seed singles (pre-registration § 9); a "
                f"mislabeled repeat would inflate sigma_total")


def _resolve_tier(arms_present: set[str], has_anchor: bool) -> str:
    has_a = "A" in arms_present
    has_b = "B" in arms_present
    if has_a and has_b:
        return "full"
    if has_a and not has_b:
        return "run_only"
    if has_b and not has_a:
        return "total_with_anchor" if has_anchor else "total_only"
    return "empty"


def _values(replicates: Sequence[Replicate], metric: str) -> list[float]:
    return [r.metrics[metric] for r in replicates if r.metrics[metric] is not None]


def analyze(units: Sequence[UnitSpec], thresholds: dict[str, float], *,
            coldstart_k: int = DEFAULT_COLDSTART_K,
            bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
            bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
            target_fpr: float = DEFAULT_TARGET_FPR,
            expected_scenario: str = DEFAULT_SCENARIO,
            sigma_run_anchors: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """Reduce Arm A and Arm B JOINTLY into per-(defense x metric) mean/SD/CI
    summaries and a variance decomposition. Never errors on a missing arm.

    ``sigma_run_anchors`` maps defense -> external sigma_run (EXP-005c/005e
    same-seed pair); it is applied PER DEFENSE to the PRIMARY metric only, in
    the Arm-B-only tier. A defense absent from the mapping stays total_only."""
    # fail fast on manifest-shape preconditions (duplicates + arm definitions)
    _validate_units(units)

    # validate every referenced defense up-front (fail fast at the boundary)
    for u in units:
        if u.defense not in DEFENSE_SCORE_SPECS:
            raise VarianceEnvelopeError(
                f"unknown defense {u.defense!r}; known: {sorted(DEFENSE_SCORE_SPECS)}")
        if u.defense not in thresholds:
            raise VarianceEnvelopeError(
                f"no provisional threshold supplied for defense {u.defense!r} "
                f"(add it to --thresholds); refusing to derive one from the logs")

    # anchor keys must name known defenses (fail-fast posture)
    for d in (sigma_run_anchors or {}):
        if d not in DEFENSE_SCORE_SPECS:
            raise VarianceEnvelopeError(
                f"--sigma-run-anchors names unknown defense {d!r}; "
                f"known: {sorted(DEFENSE_SCORE_SPECS)}")

    expected_stem = _scenario_stem(expected_scenario)
    replicates: list[Replicate] = [
        reduce_unit(u, DEFENSE_SCORE_SPECS[u.defense], thresholds[u.defense],
                    k=coldstart_k, expected_scenario_stem=expected_stem)
        for u in units
    ]

    defenses = sorted({r.defense for r in replicates})
    summary: dict[str, Any] = {}
    decomposition: dict[str, Any] = {}

    for defense in defenses:
        d_reps = [r for r in replicates if r.defense == defense]
        by_arm = {arm: [r for r in d_reps if r.arm == arm] for arm in ARMS}
        arms_present = {arm for arm in ARMS if by_arm[arm]}

        # per (arm x metric) mean/SD/CI. A deterministic per-cell RNG seed keeps
        # the whole report reproducible while decorrelating cells.
        d_i = defenses.index(defense)
        summary[defense] = {}
        for arm in ARMS:
            if not by_arm[arm]:
                continue
            a_i = ARMS.index(arm)
            summary[defense][arm] = {}
            for m_i, metric in enumerate(METRICS):
                # deterministic per-cell RNG seed (no string hashing) so the
                # whole report is reproducible across processes while cells
                # stay decorrelated
                cell_seed = bootstrap_seed + 911 * d_i + 37 * a_i + 7 * m_i
                stat = mean_sd_ci(_values(by_arm[arm], metric),
                                  bootstrap_n=bootstrap_n, rng_seed=cell_seed)
                summary[defense][arm][metric] = stat.as_dict()

        # per-metric decomposition combining arms
        d_anchor = (sigma_run_anchors or {}).get(defense)  # per-defense anchor
        decomposition[defense] = {}
        for metric in METRICS:
            sigma_run = _sd_or_none(summary[defense].get("A", {}).get(metric))
            sigma_total = _sd_or_none(summary[defense].get("B", {}).get(metric))
            run_source: Optional[str] = "Arm A repeats" if sigma_run is not None else None

            # this defense's anchor applies to the PRIMARY metric only
            has_anchor = (metric == PRIMARY_METRIC and d_anchor is not None)

            # Arm-B-only: the per-defense anchor resolves seed variance
            if sigma_run is None and sigma_total is not None and has_anchor:
                sigma_run = float(d_anchor)
                run_source = "external per-defense --sigma-run-anchors (EXP-005c/005e pair)"

            tier = _resolve_tier(arms_present, has_anchor)
            dec = decompose(sigma_run, sigma_total, tier=tier,
                            sigma_run_source=run_source)
            decomposition[defense][metric] = dec.as_dict()

    meta = {
        "study": "per-defense variance envelope",
        "scenario_scope": expected_stem,  # validated registered-scenario stem
        "primary_metric": PRIMARY_METRIC,
        "metrics": list(METRICS),
        "coldstart_k": coldstart_k,
        "target_fpr": target_fpr,
        "bootstrap_n": bootstrap_n,
        "bootstrap_seed": bootstrap_seed,
        "thresholds": thresholds,
        "sigma_run_anchors": dict(sigma_run_anchors) if sigma_run_anchors else None,
        "n_units": len(units),
        "arms_present": sorted({r.arm for r in replicates}),
        "defenses": defenses,
        "defense_score_specs": {
            d: {"score_field": s.score_field,
                "higher_is_trust": s.higher_is_trust, "source": s.source,
                "result_config": s.result_config,
                "row_defense_token": s.row_defense_token}
            for d, s in DEFENSE_SCORE_SPECS.items()},
        "threshold_provenance": (
            "FIXED provisional per-defense cut, applied unchanged to every "
            "replicate; never derived from the analysed logs"),
    }

    return {
        "meta": meta,
        "replicates": [_replicate_dict(r) for r in replicates],
        "summary": summary,
        "decomposition": decomposition,
    }


def _sd_or_none(cell: Optional[dict]) -> Optional[float]:
    if not cell:
        return None
    return cell.get("sd")


def _replicate_dict(r: Replicate) -> dict[str, Any]:
    return {
        "arm": r.arm,
        "defense": r.defense,
        "seed": r.seed,
        "replicate": r.replicate,
        "metrics": dict(r.metrics),
        "diagnostics": dict(r.diagnostics),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def format_report(report: dict[str, Any]) -> str:
    meta = report["meta"]
    lines: list[str] = [
        "# Per-defense variance envelope",
        f"scenario={meta['scenario_scope']}  coldstart_k={meta['coldstart_k']}  "
        f"target_fpr={meta['target_fpr']}  bootstrap_n={meta['bootstrap_n']}  "
        f"arms={','.join(meta['arms_present']) or 'none'}",
        f"primary_metric={meta['primary_metric']} (fixed provisional threshold, "
        f"never re-derived)",
    ]
    if meta.get("sigma_run_anchors"):
        lines.append(
            "sigma_run_anchors (per-defense, primary metric only): "
            f"{meta['sigma_run_anchors']}")
    lines.append("")
    for defense in meta["defenses"]:
        cut = meta["thresholds"].get(defense)
        spec = meta["defense_score_specs"][defense]
        lines.append(f"## {defense}  (field={spec['score_field']} "
                     f"trust={spec['higher_is_trust']} cut={cut})")
        lines.append("| metric | arm | mean | sd | 95% CI | n |")
        lines.append("|---|---|---|---|---|---|")
        for metric in meta["metrics"]:
            for arm in ARMS:
                cell = report["summary"].get(defense, {}).get(arm, {}).get(metric)
                if not cell:
                    continue
                ci = f"[{_fmt(cell['ci_low'])}, {_fmt(cell['ci_high'])}]"
                lines.append(f"| {metric} | {arm} | {_fmt(cell['mean'])} | "
                             f"{_fmt(cell['sd'])} | {ci} | {cell['n']} |")
        lines.append("")
        lines.append("| metric | tier | sigma_run | sigma_total | sigma_seed | note |")
        lines.append("|---|---|---|---|---|---|")
        for metric in meta["metrics"]:
            d = report["decomposition"][defense][metric]
            lines.append(
                f"| {metric} | {d['tier']} | {_fmt(d['sigma_run'])} | "
                f"{_fmt(d['sigma_total'])} | {_fmt(d['sigma_seed'])} | "
                f"{d['note'] or ''} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# orchestration + CLI
# ---------------------------------------------------------------------------

def run_from_paths(manifest_path: Path, thresholds_path: Path, *,
                   out_path: Optional[Path] = None,
                   coldstart_k: int = DEFAULT_COLDSTART_K,
                   bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
                   bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
                   target_fpr: float = DEFAULT_TARGET_FPR,
                   expected_scenario: str = DEFAULT_SCENARIO,
                   sigma_run_anchors_path: Optional[Path] = None) -> dict[str, Any]:
    units = load_manifest(Path(manifest_path))
    thresholds = load_thresholds(Path(thresholds_path))
    sigma_run_anchors = (load_sigma_run_anchors(Path(sigma_run_anchors_path))
                         if sigma_run_anchors_path is not None else None)
    report = analyze(units, thresholds, coldstart_k=coldstart_k,
                     bootstrap_n=bootstrap_n, bootstrap_seed=bootstrap_seed,
                     target_fpr=target_fpr, expected_scenario=expected_scenario,
                     sigma_run_anchors=sigma_run_anchors)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False: never write NaN/Infinity to the artifact (invalid
        # per RFC 8259; breaks strict downstream parsers). Surfaces loudly if a
        # non-finite ever slips through instead of writing corrupt JSON.
        out_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    # allow_abbrev=False so the removed scalar --sigma-run-anchor is a hard
    # error rather than a silent prefix match onto --sigma-run-anchors
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     allow_abbrev=False)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSON with a 'units' list (arm, defense, seed, "
                             "replicate, result, signal)")
    parser.add_argument("--thresholds", required=True, type=Path,
                        help="JSON mapping defense -> FIXED provisional cut")
    parser.add_argument("--out", type=Path, default=None,
                        help="machine-readable JSON output path")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                        help="registered scenario every unit must match (stem-"
                             "normalized against both inputs; this study is S4-only)")
    parser.add_argument("--coldstart-k", type=int, default=DEFAULT_COLDSTART_K)
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR,
                        help="documents the FPR the provisional cut targets; "
                             "NOT used to derive any threshold")
    parser.add_argument("--sigma-run-anchors", type=Path, default=None,
                        help="JSON mapping defense -> external sigma_run "
                             "(per-defense; primary metric only, when Arm A is "
                             "absent; EXP-005c/005e pair)")
    args = parser.parse_args(argv)

    report = run_from_paths(
        args.manifest, args.thresholds, out_path=args.out,
        coldstart_k=args.coldstart_k, bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed, target_fpr=args.target_fpr,
        expected_scenario=args.scenario,
        sigma_run_anchors_path=args.sigma_run_anchors)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
