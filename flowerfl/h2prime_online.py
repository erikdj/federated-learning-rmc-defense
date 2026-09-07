"""Online H2' serving detector for the first stage of the H4 chain.

Public contracts
----------------
* `docs/reproduction/experiments.md` describes the H2' and H4 reproduction
  workflow. The detector is one GBDT with a frozen nine-feature order, trained
  on the complete H2' confirmatory corpus, with per-scenario cuts targeting
  FPR 0.10. The serving-bundle manifest SHA-256 is exported in unit custody;
  runtime code does not select model parameters or cuts.
* The bundle in `data/h4_serving/` contains `model.joblib`, `cuts.json`,
  `features.json`, and `manifest.json`; `bundle_sha256` is the SHA-256 of the
  manifest bytes.
* Online per-round feature rows must be bit-exact with
  `reproduction/protocol/h2prime/revalidate_v115.py::derive_window_feats`.
  The serving implementation keeps that construction local, while
  `tests/test_h2prime_online_parity.py` and `tests/test_window_feats_golden.py`
  compare it with the public frozen helper and recorded golden fixture. A
  parity failure blocks launch.

Flag semantics
--------------
The detector emits P(malicious); the cut is
``numpy.quantile(honest_scores, 1 - 0.10)`` (NumPy default linear
interpolation) and the flag decision is STRICT ``score > cut`` — a score
exactly equal to the cut is NOT flagged. This mirrors the frozen offline
scorer byte-for-byte (`revalidate_v115.py::flagged` with
``higher_is_trust=False``). The strict inequality is the operative contract.

Chain position
--------------
ALWAYS FIRST in the plugin chain (frozen § 7c-bis order): the detector scores
the PRE-filter signal stream — every participating client of the round, before
any FP or aggregator layer acts. `PluggableStrategy` hands the first plugin the
complete (canonicalized) results list, which is the same population and order
`ScenarioStrategy._maybe_log_signals` logs, so the online raw features are the
same numbers the offline corpus was built from.

Nothing here reads sealed material or prints seed values.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flowerfl.byzantine_defense import ByzantineDefensePlugin
from flowerfl.signal_logger import compute_per_client_signals, flatten_parameters

#: v1.15 § 2.2b trailing-window length (frozen; matches the offline WINDOW).
WINDOW: int = 3

#: The four § 2.2b derived features (frozen names, offline DERIVED order).
DERIVED_FEATS: Tuple[str, ...] = (
    "norm_variance", "loss_slope", "cos_drift", "cos_variance",
)

#: Scenario tokens the § 7.1 cuts are keyed by.
SCENARIO_TOKENS: Tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4")

#: The FROZEN v1.15 § 2.2b nine-feature order — the model's training column
#: order, transcribed verbatim from the committed serving bundle's
#: `data/h4_serving/features.json` (itself echoed from
#: `H2PRIME_ADJUDICATION.json` `_meta.features_frozen_order`).
#: The feature list `features.json` must EQUAL this exactly (names AND order)
#: at load — a bundle with reordered or renamed features would otherwise pass
#: the shape checks and scoring would silently permute columns relative to
#: the GBDT's training order.
FROZEN_FEATURE_ORDER: Tuple[str, ...] = (
    "update_norm", "train_loss", "num_examples",
    "norm_variance", "loss_slope",
    "cos_to_median", "L2_to_median", "cos_drift", "cos_variance",
)

#: The three bundle payload files whose sha256 the manifest must pin.
BUNDLE_FILES: Tuple[str, ...] = ("model.joblib", "cuts.json", "features.json")

#: --- Erratum B (2026-08-18, RULED, methodology v1.53) -----------------------
#: Bundle v2: cuts per (scenario, arm-class), calibrated on observe-only
#: population-matched honest scores. The v2 payload shares `model.joblib` and
#: `features.json` with v1 and adds `cuts_v2.json` + `manifest_v2.json`;
#: bundle_v2_sha256 = sha256 of the manifest_v2.json bytes.
BUNDLE_FILES_V2: Tuple[str, ...] = (
    "model.joblib", "cuts_v2.json", "features.json",
)

#: The closed cuts-version set. Selection is EXPLICIT (pinned in the unit
#: config / builder CLI) — presence of v2 files on disk NEVER auto-selects.
CUTS_VERSIONS: Tuple[str, ...] = ("v1", "v2")

#: v2 scenario tokens: C0 carries its OWN calibrated cut (the erratum-A
#: E3-bis C0->S0 alias is RETIRED on the v2 path; it survives only for v1).
SCENARIO_TOKENS_V2: Tuple[str, ...] = ("C0", "S0", "S1", "S2", "S3", "S4")

#: Erratum-B arm-classes — the aggregator family that shapes the model state
#: the detector scores against (§ B1 cut grain).
ARM_CLASSES: Tuple[str, ...] = ("krum_family", "ts_family", "fedavg_family")

#: strategy class name -> arm-class. Covers the five H4 composition arms AND
#: the observer-attached BASE arms the § B1 calibration units run (the
#: observer rides Krum / TrustScore / FedAvg so each unit's model state is
#: the deployment-adjacent state of its arm-class).
ARM_CLASS_BY_STRATEGY: Dict[str, str] = {
    "ScenarioH2PFPKrum": "krum_family",
    "ScenarioH2PKrum": "krum_family",
    "ScenarioH2PFPTS": "ts_family",
    "ScenarioH2PTS": "ts_family",
    "ScenarioH2PFP": "fedavg_family",
    "ScenarioKrum": "krum_family",
    "ScenarioTrustScore": "ts_family",
    "ScenarioNone": "fedavg_family",
}

_SCENARIO_NAME_RE = re.compile(r"^[sS](\d)_")
_C0_STEM = "c0_clean_no_attack"

#: Erratum-A E3-bis (methodology v1.52): EXPLICIT scenario -> cut aliases for
#: scenarios outside the S0-S4 calibration corpus. `C0_clean_no_attack` uses
#: the S0 cut — S0 is the confounder-free population whose honest rows are the
#: closest available calibration reference; C0 is S0 minus its attackers. This
#: is a DECLARED mapping resolved loudly, never a silent .get()-with-fallback;
#: an unknown scenario with no cut and no declared alias still refuses.
SERVING_CUT_ALIASES: Dict[str, str] = {
    "C0_clean_no_attack": "S0",
}


class H2PrimeBundleError(RuntimeError):
    """The serving bundle is missing, malformed, or fails hash verification.

    Always a refusal-to-run: the § 7.1 instrument is frozen, so a bundle that
    cannot be verified must never be scored around or defaulted past.
    """


def scenario_token_from_name(scenario_name: str, cuts_version: str = "v1") -> str:
    """`S3_identity_reset_only` / `s3_identity_reset_only` -> ``"S3"``.

    v1 (default): scenarios outside S0-S4 resolve ONLY through the explicit
    `SERVING_CUT_ALIASES` declaration (erratum-A E3-bis: C0 -> the S0 cut),
    announced loudly at resolution.

    v2 (erratum B): the alias table is RETIRED — `C0_clean_no_attack`
    resolves to its OWN ``"C0"`` token (bundle v2 carries a calibrated C0
    cut). Fail-loud on anything else: the per-(scenario, arm-class) cut is
    the operating point, and guessing a scenario would silently move it.
    """
    _require_cuts_version(cuts_version)
    stem = str(scenario_name)
    if cuts_version == "v2":
        if stem.lower() == _C0_STEM:
            return "C0"
        match = _SCENARIO_NAME_RE.match(stem)
        if not match or f"S{match.group(1)}" not in SCENARIO_TOKENS:
            raise H2PrimeBundleError(
                f"cannot derive a cuts_v2 scenario token from scenario name "
                f"{scenario_name!r}; the v2 cut table is keyed "
                f"{list(SCENARIO_TOKENS_V2)} (erratum B — the E3-bis alias "
                f"table is retired for v2) and no default is permissible."
            )
        return f"S{match.group(1)}"
    if stem in SERVING_CUT_ALIASES:
        token = SERVING_CUT_ALIASES[stem]
        print(
            f"[H2PrimeDetector] scenario {stem!r} has no own serving cut; "
            f"using the DECLARED alias -> {token} cut (erratum-A E3-bis).",
            flush=True,
        )
        return token
    match = _SCENARIO_NAME_RE.match(stem)
    if not match:
        raise H2PrimeBundleError(
            f"cannot derive an S0-S4 scenario token from scenario name "
            f"{scenario_name!r} and no explicit serving-cut alias is declared "
            f"for it (declared aliases: {sorted(SERVING_CUT_ALIASES)}); the "
            f"H2' serving cuts are per-scenario (spec 2026-08-16 § 7.1, "
            f"erratum-A E3-bis) and no default is permissible."
        )
    token = f"S{match.group(1)}"
    if token not in SCENARIO_TOKENS:
        raise H2PrimeBundleError(
            f"scenario token {token!r} (from {scenario_name!r}) is outside "
            f"the frozen cut table {SCENARIO_TOKENS}."
        )
    return token


def flag_scores(scores, cut: float) -> np.ndarray:
    """The frozen flag decision: STRICT ``score > cut``.

    Mirrors `revalidate_v115.py::flagged(scores, cut, higher_is_trust=False)`
    byte-for-byte — a score exactly equal to the cut is NOT flagged. Kept as a
    module-level function so the boundary test pins the inequality directly.
    """
    return np.asarray(scores, dtype=float) > float(cut)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_cuts_version(cuts_version: str) -> str:
    """Validate the EXPLICIT cuts-version selection (erratum B: pinned in the
    unit config, never auto-detected from what exists on disk)."""
    version = str(cuts_version)
    if version not in CUTS_VERSIONS:
        raise H2PrimeBundleError(
            f"unknown cuts-version {cuts_version!r}; the closed set is "
            f"{list(CUTS_VERSIONS)} (erratum B § B2 — explicit selection, "
            f"no auto-detect, no default beyond the incumbent 'v1')."
        )
    return version


@dataclass(frozen=True)
class ServingBundle:
    """The verified serving instrument (§ 7.1 v1, or erratum-B v2), loaded
    once at server startup."""

    bundle_dir: Path
    model: Any
    #: v1: {scenario_token: cut}. v2: {scenario_token: {arm_class: cut}} —
    #: resolve through `cut_for`, never by direct subscript.
    cuts: Dict[str, Any]
    features: Tuple[str, ...]
    manifest: Dict[str, Any]
    #: sha256 of the manifest BYTES (manifest.json for v1, manifest_v2.json
    #: for v2) — the single custody pin (BUILD_CONTRACT: goes into unit
    #: custody, the smoke EXP doc, MLflow).
    bundle_sha256: str
    #: Which cut table this bundle serves — "v1" (per-scenario, § 7.1) or
    #: "v2" (per-(scenario, arm-class), erratum B). Default keeps every
    #: pre-erratum constructor call byte-identical.
    cuts_version: str = "v1"

    def cut_for(self, scenario_token: str, arm_class: Optional[str] = None) -> float:
        """The operating point for (scenario_token[, arm_class]) — refusal on
        any unknown key, never a fallback (erratum B: no aliases on v2)."""
        if scenario_token not in self.cuts:
            raise H2PrimeBundleError(
                f"no serving cut for scenario {scenario_token!r}; bundle "
                f"carries {sorted(self.cuts)} — refusing to improvise an "
                f"operating point (spec 2026-08-16 § 7.1)."
            )
        if self.cuts_version == "v1":
            return float(self.cuts[scenario_token])
        if arm_class not in ARM_CLASSES:
            raise H2PrimeBundleError(
                f"cuts_v2 requires an arm_class from {list(ARM_CLASSES)}; "
                f"got {arm_class!r} (erratum B § B1: cuts are per "
                f"(scenario, arm-class); an unknown pair is a refusal)."
            )
        per_class = self.cuts[scenario_token]
        if arm_class not in per_class:
            raise H2PrimeBundleError(
                f"no serving cut for (scenario={scenario_token!r}, "
                f"arm_class={arm_class!r}); bundle carries "
                f"{sorted(per_class)} for that scenario — refusing "
                f"(erratum B: unknown pair = refusal, no fallback)."
            )
        return float(per_class[arm_class])


def _read_manifest_and_verify_files(
    bundle_dir: Path, manifest_name: str, bundle_files: Tuple[str, ...],
) -> Tuple[Dict[str, Any], str]:
    """Shared manifest read + per-file sha256 verification (v1 and v2)."""
    manifest_path = bundle_dir / manifest_name
    if not manifest_path.is_file():
        raise H2PrimeBundleError(
            f"serving bundle manifest missing: {manifest_path} — the H4 "
            f"detector arms cannot run without the frozen instrument."
        )
    manifest_bytes = manifest_path.read_bytes()
    bundle_sha256 = _sha256_bytes(manifest_bytes)
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError as exc:
        raise H2PrimeBundleError(
            f"serving bundle manifest is not valid JSON: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise H2PrimeBundleError(
            f"serving bundle manifest must be a JSON object, got "
            f"{type(manifest).__name__}: {manifest_path}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(bundle_files):
        raise H2PrimeBundleError(
            f"serving bundle manifest 'files' must map exactly "
            f"{sorted(bundle_files)} to sha256 hex digests; got "
            f"{sorted(files) if isinstance(files, dict) else files!r}"
        )
    for name in bundle_files:
        path = bundle_dir / name
        if not path.is_file():
            raise H2PrimeBundleError(f"serving bundle file missing: {path}")
        recorded = str(files[name])
        actual = _sha256_bytes(path.read_bytes())
        if actual != recorded:
            raise H2PrimeBundleError(
                f"serving bundle file {name} fails sha256 verification: "
                f"recorded {recorded}, actual {actual}. The frozen "
                f"instrument has drifted — refusing to serve it."
            )
    return manifest, bundle_sha256


def _finite_cut(value: Any, where: str) -> float:
    try:
        cut = float(value)
    except (TypeError, ValueError):
        cut = float("nan")
    if not np.isfinite(cut):
        raise H2PrimeBundleError(f"{where} is not a finite float: {value!r}")
    return cut


def _load_cuts_v1(bundle_dir: Path) -> Dict[str, float]:
    """cuts.json — exactly the five per-scenario cuts, finite floats."""
    cuts_raw = json.loads((bundle_dir / "cuts.json").read_bytes())
    if not isinstance(cuts_raw, dict) or set(cuts_raw) != set(SCENARIO_TOKENS):
        raise H2PrimeBundleError(
            f"cuts.json must map exactly {list(SCENARIO_TOKENS)} to floats; "
            f"got keys {sorted(cuts_raw) if isinstance(cuts_raw, dict) else cuts_raw!r}. "
            f"A single global cut is the § 2.2-rejected degeneracy and is not "
            f"a permissible fallback (spec 2026-08-16 § 7.1)."
        )
    return {
        token: _finite_cut(value, f"cuts.json[{token!r}]")
        for token, value in cuts_raw.items()
    }


def _load_cuts_v2(bundle_dir: Path) -> Dict[str, Dict[str, float]]:
    """cuts_v2.json — EXACTLY {C0,S0..S4} x {the three arm-classes}, finite
    floats (erratum B § B2: 18 cuts, C0 calibrated directly, no aliases)."""
    cuts_raw = json.loads((bundle_dir / "cuts_v2.json").read_bytes())
    if (not isinstance(cuts_raw, dict)
            or set(cuts_raw) != set(SCENARIO_TOKENS_V2)):
        raise H2PrimeBundleError(
            f"cuts_v2.json must map exactly {list(SCENARIO_TOKENS_V2)} to "
            f"per-arm-class cut maps; got keys "
            f"{sorted(cuts_raw) if isinstance(cuts_raw, dict) else cuts_raw!r} "
            f"(erratum B § B2 — C0 must carry its OWN calibrated cut)."
        )
    cuts: Dict[str, Dict[str, float]] = {}
    for token, per_class_raw in cuts_raw.items():
        if (not isinstance(per_class_raw, dict)
                or set(per_class_raw) != set(ARM_CLASSES)):
            raise H2PrimeBundleError(
                f"cuts_v2.json[{token!r}] must map exactly "
                f"{list(ARM_CLASSES)} to floats; got "
                f"{sorted(per_class_raw) if isinstance(per_class_raw, dict) else per_class_raw!r} "
                f"(erratum B § B1: the cut grain is (scenario, arm-class))."
            )
        cuts[token] = {
            arm_class: _finite_cut(
                value, f"cuts_v2.json[{token!r}][{arm_class!r}]"
            )
            for arm_class, value in per_class_raw.items()
        }
    return cuts


def load_serving_bundle(
    bundle_dir: "Path | str", cuts_version: str = "v1",
) -> ServingBundle:
    """Load and VERIFY the serving bundle; refuse loudly on any defect.

    ``cuts_version`` selects the cut table EXPLICITLY (erratum B § B2:
    pinned in the unit config, NEVER auto-detected from what exists on
    disk): ``"v1"`` (default — byte-identical to the pre-erratum loader)
    reads `manifest.json`/`cuts.json`; ``"v2"`` reads `manifest_v2.json`/
    `cuts_v2.json` keyed {scenario: {arm_class: cut}}. In both versions the
    manifest carries a ``files`` map with the sha256 of each payload file;
    every recorded hash is recomputed from the bytes on disk and must match
    exactly. The returned ``bundle_sha256`` is the sha256 of the ACTIVE
    manifest's bytes (bundle_v2_sha256 for v2).
    """
    version = _require_cuts_version(cuts_version)
    bundle_dir = Path(bundle_dir)
    if version == "v2":
        manifest, bundle_sha256 = _read_manifest_and_verify_files(
            bundle_dir, "manifest_v2.json", BUNDLE_FILES_V2
        )
        cuts: Dict[str, Any] = _load_cuts_v2(bundle_dir)
    else:
        manifest, bundle_sha256 = _read_manifest_and_verify_files(
            bundle_dir, "manifest.json", BUNDLE_FILES
        )
        cuts = _load_cuts_v1(bundle_dir)

    # features.json — must EQUAL the frozen nine-name order exactly (names
    # AND order). The feature list is the model's training column order; any
    # reordering or renaming that still passed a shape check would silently
    # permute the design matrix relative to the GBDT.
    features_raw = json.loads((bundle_dir / "features.json").read_bytes())
    if (
        not isinstance(features_raw, list)
        or tuple(features_raw) != FROZEN_FEATURE_ORDER
    ):
        raise H2PrimeBundleError(
            f"features.json must equal the frozen feature order EXACTLY "
            f"(names and order): expected {list(FROZEN_FEATURE_ORDER)}, got "
            f"{features_raw!r}. A reordered or renamed feature list would "
            f"silently permute the model's input columns — refusing to serve."
        )
    features = tuple(features_raw)

    import joblib

    try:
        model = joblib.load(bundle_dir / "model.joblib")
    except Exception as exc:  # noqa: BLE001 - any load failure is a refusal
        raise H2PrimeBundleError(
            f"model.joblib failed to load: {exc}"
        ) from exc
    if not callable(getattr(model, "predict_proba", None)):
        raise H2PrimeBundleError(
            f"model.joblib does not expose predict_proba "
            f"(got {type(model).__name__}) — not a servable classifier."
        )
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(features):
        raise H2PrimeBundleError(
            f"model expects {int(n_in)} features but features.json names "
            f"{len(features)} — the bundle is internally inconsistent."
        )

    return ServingBundle(
        bundle_dir=bundle_dir,
        model=model,
        cuts=cuts,
        features=features,
        manifest=manifest,
        bundle_sha256=bundle_sha256,
        cuts_version=version,
    )


class OnlineWindowFeatureBuilder:
    """Incremental § 2.2b window-feature derivation, bit-exact with the frozen
    offline `derive_window_feats`.

    The construction is transcribed VERBATIM from the frozen builder (see the
    module docstring for why it cannot be imported in-image): per
    (logical_cid, tenure-episode), a trailing window of <= WINDOW rows using
    PAST+CURRENT rows only; a tenure value <= the previous row's tenure starts
    a fresh episode; identical numpy calls (`np.var` ddof=0, `np.polyfit` deg
    1, plain float subtraction) on identically-filtered value lists. The
    committed golden parity test replays the recorded fixture through BOTH
    builders and asserts bit-exact equality — that test, not this docstring,
    is the § 7 item-2 launch gate.
    """

    def __init__(self) -> None:
        self._episodes: Dict[str, List[Dict[str, Any]]] = {}
        self._prev_tenure: Dict[str, Optional[int]] = {}

    def add_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Consume one raw per-(client, round) row; return a NEW row dict with
        the four derived features added. The input row is not mutated."""
        cid = row["logical_cid"]
        t = row.get("tenure")
        prev_tenure = self._prev_tenure.get(cid)
        episode = self._episodes.get(cid, [])
        if prev_tenure is not None and t is not None and t <= prev_tenure:
            episode = []                       # rejoin -> new episode (frozen)
        self._prev_tenure[cid] = t
        episode = [*episode, dict(row)]        # new list + defensive row copy
        self._episodes[cid] = episode

        w = episode[-WINDOW:]
        norms = [x["update_norm"] for x in w if x.get("update_norm") is not None]
        losses = [x["train_loss"] for x in w if x.get("train_loss") is not None]
        coss = [x["cos_to_median"] for x in w if x.get("cos_to_median") is not None]
        derived: Dict[str, Any] = {}
        derived["norm_variance"] = float(np.var(norms)) if len(norms) > 1 else 0.0
        derived["cos_variance"] = float(np.var(coss)) if len(coss) > 1 else 0.0
        if len(losses) > 1:
            xs = np.arange(len(losses), dtype=float)
            derived["loss_slope"] = float(np.polyfit(xs, np.asarray(losses), 1)[0])
        else:
            derived["loss_slope"] = 0.0
        derived["cos_drift"] = (
            float(coss[-1] - coss[-2]) if len(coss) > 1 else 0.0
        )
        return {**row, **derived}


class OnlineH2PrimeDetectorPlugin(ByzantineDefensePlugin):
    """§ 7c-bis step 1: score EVERY participating client on the pre-filter
    stream; flagged clients are dropped from this round's aggregation.

    Placed FIRST in every detector-bearing arm's chain, so `score_updates`
    receives the complete canonicalized cohort — the same population, in the
    same order, that the signal logger writes (raw features are computed by
    the SAME `compute_per_client_signals` the logger uses, so the online
    feature rows cannot drift from the offline corpus construction).

    Window state accumulates for every scored client every round — a client
    the detector flags is dropped from THIS round's aggregation but keeps
    participating per the scenario schedule, exactly as its signal-log rows
    keep accumulating offline.
    """

    def __init__(
        self,
        bundle: ServingBundle,
        scenario_token: str,
        round_offset: int = 1,
        observe_only: bool = False,
        arm_class: Optional[str] = None,
    ) -> None:
        """
        Args:
            bundle: a VERIFIED `ServingBundle` (see `load_serving_bundle`).
            scenario_token: which scenario cut governs this run (v1: S0-S4;
                v2: C0 + S0-S4 — erratum B, no aliases).
            round_offset: server_round -> scenario_round offset. 1 for
                ScenarioStrategy (its discovery round shifts the schedule).
            observe_only: erratum-B § B1 calibration mode — score and LOG
                every participating client exactly as the enforcing mode
                does, but `filter_updates` drops NOBODY. Default False =
                byte-identical enforcing behavior.
            arm_class: the erratum-B arm-class (see ARM_CLASS_BY_STRATEGY).
                REQUIRED for a v2 bundle (the cut grain); optional metadata
                for v1 (recorded into observe rows/custody when known).
        """
        if bundle.cuts_version == "v2" and arm_class not in ARM_CLASSES:
            raise H2PrimeBundleError(
                f"a cuts_v2 bundle requires an arm_class from "
                f"{list(ARM_CLASSES)}; got {arm_class!r} — the erratum-B "
                f"cut grain is (scenario, arm-class) and no default is "
                f"permissible."
            )
        if arm_class is not None and arm_class not in ARM_CLASSES:
            raise H2PrimeBundleError(
                f"unknown arm_class {arm_class!r}; the closed set is "
                f"{list(ARM_CLASSES)} (erratum B § B1)."
            )
        self._bundle = bundle
        self._scenario_token = str(scenario_token)
        self._cut = bundle.cut_for(self._scenario_token, arm_class)
        self._observe_only = bool(observe_only)
        self._arm_class = arm_class
        #: Erratum-B calibration log: one row per (client, round) scored in
        #: observe-only mode — raw score + would-flag at the pinned cut. The
        #: runner exports these (ground-truth-enriched) into the unit's
        #: result JSON `h2p_observe` block. Always empty in enforcing mode.
        self._observe_rows: List[Dict[str, Any]] = []
        self._round_offset = int(round_offset)
        self._builder = OnlineWindowFeatureBuilder()
        # Same tenure semantics as the strategy's ground-truth layer
        # (scenario_strategy.compute_tenure): a mutable {logical_cid ->
        # first_round} cache over the identical participation sequence.
        self._tenure_first_seen: Dict[str, int] = {}
        # Per-round {idx: 1.0 keep / 0.0 drop} (ByzantineDefensePlugin
        # convention) plus the continuous P(malicious) per logical identity.
        self._round_scores: Dict[int, Dict[int, float]] = {}
        self._probabilities_by_round: Dict[int, Dict[str, float]] = {}
        self._flagged_by_round: Dict[int, Tuple[str, ...]] = {}

    @property
    def name(self) -> str:
        return "H2PrimeDetector"

    @property
    def bundle_sha256(self) -> str:
        """The custody pin exported as `serving_bundle_sha256` per unit."""
        return self._bundle.bundle_sha256

    @property
    def cut(self) -> float:
        return self._cut

    @property
    def scenario_token(self) -> str:
        return self._scenario_token

    @property
    def observe_only(self) -> bool:
        return self._observe_only

    @property
    def arm_class(self) -> Optional[str]:
        return self._arm_class

    @property
    def cuts_version(self) -> str:
        return self._bundle.cuts_version

    @property
    def observe_rows(self) -> List[Dict[str, Any]]:
        """Defensive copies of the accumulated observe-only calibration rows."""
        return [dict(row) for row in self._observe_rows]

    def flagged_identities(self, server_round: int) -> Tuple[str, ...]:
        return self._flagged_by_round.get(int(server_round), ())

    # -- scoring ------------------------------------------------------------

    def _logical_identity(self, cid: str, server_round: int) -> str:
        identity = self.resolve_identity(str(cid))
        if identity is None:
            raise RuntimeError(
                f"[H2PrimeDetector] round {server_round}: client cid={cid!r} "
                f"has no logical identity mapping — the frozen § 7c-bis order "
                f"requires the detector to score EVERY participating client, "
                f"and an unmapped client cannot be windowed by identity. "
                f"Refusing to score around it (v1.17 misattribution family)."
            )
        return str(identity)

    def score_updates(self, results, server_round: int) -> Dict[int, float]:
        if not results:
            return {}
        from flowerfl.scenario_strategy import compute_tenure

        scenario_round = int(server_round) - self._round_offset

        # Raw per-round features: the SAME construction the signal logger
        # uses (_maybe_log_signals), on the same pre-filter results order.
        logicals: List[str] = []
        flat_updates: List[np.ndarray] = []
        train_losses: List[Optional[float]] = []
        num_examples_list: List[int] = []
        for client_proxy, fit_res in results:
            logicals.append(
                self._logical_identity(str(client_proxy.cid), server_round)
            )
            flat_updates.append(flatten_parameters(fit_res.parameters))
            tl = (getattr(fit_res, "metrics", None) or {}).get("train_loss")
            try:
                train_losses.append(float(tl) if tl is not None else None)
            except (TypeError, ValueError):
                train_losses.append(None)
            num_examples_list.append(int(fit_res.num_examples))

        signals = compute_per_client_signals(
            flat_updates, train_losses, num_examples_list
        )

        vectors: List[List[float]] = []
        for idx, logical in enumerate(logicals):
            tenure = compute_tenure(
                logical, int(server_round), self._tenure_first_seen
            )
            raw_row = {
                "logical_cid": logical,
                "scenario_round": scenario_round,
                "tenure": tenure,
                **signals[idx],
            }
            row = self._builder.add_row(raw_row)
            vector: List[float] = []
            for feat in self._bundle.features:
                value = row.get(feat)
                if value is None or not np.isfinite(float(value)):
                    raise RuntimeError(
                        f"[H2PrimeDetector] round {server_round} client "
                        f"{logical!r}: feature {feat!r} is "
                        f"missing/non-finite ({value!r}) — the frozen "
                        f"feature contract admits no imputation."
                    )
                vector.append(float(value))
            vectors.append(vector)

        probabilities = self._bundle.model.predict_proba(
            np.asarray(vectors, dtype=float)
        )[:, 1]
        flags = flag_scores(probabilities, self._cut)   # STRICT score > cut

        self._probabilities_by_round[int(server_round)] = {
            logicals[idx]: float(probabilities[idx])
            for idx in range(len(results))
        }

        if self._observe_only:
            # Erratum-B § B1: the SAME scoring, the SAME would-flag decision
            # at the pinned cut — logged, never enforced. Everyone keeps
            # score 1.0 and `flagged_identities` stays empty (it records
            # ENFORCED drops; would-flags live in the observe rows).
            self._observe_rows.extend(
                {
                    "server_round": int(server_round),
                    "scenario_round": int(scenario_round),
                    "logical_cid": logicals[idx],
                    "score": float(probabilities[idx]),
                    "would_flag": bool(flags[idx]),
                }
                for idx in range(len(results))
            )
            scores = {idx: 1.0 for idx in range(len(results))}
            self._round_scores[int(server_round)] = dict(scores)
            self._flagged_by_round[int(server_round)] = ()
            return scores

        scores = {
            idx: (0.0 if bool(flags[idx]) else 1.0)
            for idx in range(len(results))
        }
        self._round_scores[int(server_round)] = dict(scores)
        self._flagged_by_round[int(server_round)] = tuple(
            logicals[idx] for idx in range(len(results)) if bool(flags[idx])
        )
        return scores

    def filter_updates(self, results, scores, threshold: float = 0.0):
        """Drop flagged clients (score 0.0) from this round's aggregation.

        The decision was made in `score_updates` at the frozen cut; this
        method only enforces it, with a loud per-round record. In
        observe-only mode (erratum B § B1) NOBODY is dropped — the would-flag
        decisions were logged in `score_updates` and this method passes the
        complete cohort through, loudly.
        """
        if self._observe_only:
            n_would_flag = sum(
                1 for row in self._observe_rows if row["would_flag"]
            )
            print(
                f"[H2PrimeDetector] OBSERVE-ONLY round: kept "
                f"{len(results)}/{len(results)} clients (scenario="
                f"{self._scenario_token} arm_class={self._arm_class} "
                f"cut={self._cut:.6f} strict '>', dropped=none; "
                f"cumulative would-flag rows={n_would_flag})"
            )
            return list(results)
        survivors = []
        dropped = []
        for idx, (client_proxy, fit_res) in enumerate(results):
            if scores.get(idx, 1.0) > 0.0:
                survivors.append((client_proxy, fit_res))
            else:
                dropped.append(str(client_proxy.cid))
        print(
            f"[H2PrimeDetector] Round kept {len(survivors)}/{len(results)} "
            f"clients (scenario={self._scenario_token} cut={self._cut:.6f} "
            f"strict '>', dropped={dropped or 'none'})"
        )
        return survivors
