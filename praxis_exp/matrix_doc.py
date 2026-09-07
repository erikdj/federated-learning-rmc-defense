"""Parses a matrix sweep design doc (lists for defenses/scenarios/seeds)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_TOP = ("exp_id", "slug", "hypothesis", "methodology_version", "matrix", "batch")
_MATRIX = ("defenses", "scenarios", "seeds", "mode", "max_per_client", "rounds")
_BATCH = ("job_queue", "job_definition")
_PLACEHOLDER = re.compile(r"\b(TBD|TODO|<FILL>)\b")

# Optional top-level ``run_extras`` mapping: non-matrix run-config overrides that
# are a property of the EXPERIMENT (not a defense/scenario/seed axis, so they do
# NOT multiply the cross-product). Carried verbatim into the manifest meta and
# turned into runner CLI flags by docker/entrypoint.py::runner_argv. The key set
# is an explicit allowlist so a typo (or an un-plumbed knob) fails LOUDLY at
# pre-registration parse rather than silently running the incumbent on the fleet
# (the EXP-016 silent-flavor lesson). Only the SMOTE study keys are accepted for
# now; extend this tuple as new fleet-reachable run-config knobs are wired.
# Stage-F (DESIGN_STAGE_F §4/§5/§6, checklist item 7): update_match /
# weight_mode / smote_semantic_target are fleet-reachable via
# entrypoint.stage_f_argv_from_run_extras.
_RUN_EXTRAS_ALLOWED = (
    "smote_enabled", "smote_variant", "smote_target",
    "update_match", "weight_mode", "smote_semantic_target",
    # m1 leakage fix (opt-in, default OFF): fleet-reachable via
    # entrypoint.leakage_argv_from_run_extras -> runner --normalize-train-only.
    "normalize_train_only",
    # H3 post-tau-lock cohort declaration (v1.10 s5.1 D9): which LOCKED tau/Sigma
    # pair scores this run. Fleet-reachable via
    # entrypoint.fp_cohort_argv_from_run_extras -> runner --fp-cohort. Absent =
    # key never emitted (pre-lock runs and non-FP arms byte-identical to the
    # incumbent); post-lock FP-arm runs WITHOUT it are refused by server_app.
    "fp_cohort",
    # H3 registry candidate policy (methodology v1.49; corrected-instrument
    # amendment s1): flag_gated (deployed incumbent) vs identity_only
    # (enroll-everyone). Fleet-reachable via
    # entrypoint.fp_registry_policy_argv_from_run_extras -> runner
    # --fp-registry-policy. Absent = key never emitted, incumbent instrument.
    "fp_registry_policy",
    # H4 evaluation population (erratum-A E4): legacy (sampled holdout,
    # incumbent) vs sealed_test (the locked val/test manifest's TEST indices).
    # Fleet-reachable via entrypoint.eval_split_argv_from_run_extras -> runner
    # --eval-split. Absent = key never emitted, legacy evaluator byte-identical.
    # This is how the REUSED H4 arms (Krum / TrustScore / Krum+TGE+FP / FedAvg)
    # reach the sealed evaluator — the H2P+* tokens bake it in their configs.
    "eval_split",
    # Erratum B (2026-08-18, methodology v1.53) § B1: run the H2' detector /
    # observer in OBSERVE-ONLY calibration mode (scores + would-flags logged,
    # nobody dropped). Strict bool (typos refuse at parse). Fleet-reachable
    # via entrypoint.h2p_argv_from_run_extras -> runner --h2p-observe-only.
    # Absent = key never emitted, enforcing incumbent byte-identical.
    "h2p_observe_only",
    # Erratum B § B2: EXPLICIT serving cut-table selection — v1 (per-scenario
    # § 7.1 cuts, incumbent) or v2 (per-(scenario, arm-class) calibrated
    # cuts). Closed {v1, v2}; never auto-detected. Fleet-reachable via
    # entrypoint.h2p_argv_from_run_extras -> runner --h2p-cuts-version.
    # Absent = key never emitted, v1 incumbent.
    "h2p_cuts_version",
)

#: The closed value set for run_extras.h2p_cuts_version — the same set the
#: runner's --h2p-cuts-version parser and h2prime_online._require_cuts_version
#: enforce (single source by convention; drift is caught by wiring tests).
_H2P_CUTS_VERSIONS = ("v1", "v2")

#: The closed value set for run_extras.eval_split — the same set the runner's
#: --eval-split parser and server_app's dispatch enforce (single source of
#: truth by convention; a drift between them is caught by the wiring tests).
_EVAL_SPLIT_VALUES = ("legacy", "sealed_test")


class MatrixDocError(ValueError):
    """Raised when a matrix design doc fails validation."""


@dataclass
class MatrixDoc:
    exp_id: str
    slug: str
    hypothesis: str
    methodology_version: str
    defenses: list[str]
    scenarios: list[str]
    seeds: list[int]
    mode: str
    max_per_client: int
    rounds: int
    # Same-seed replicate count (GWU-44 Lane B). OPTIONAL — defaults to 1 so
    # every existing design doc (written before the axis) parses unchanged;
    # deliberately NOT in the _MATRIX required-field tuple for that reason.
    repeats: int
    job_queue: str
    job_definition: str
    body: str
    path: Path
    # Optional (defaults to {}) so every existing design doc parses unchanged.
    run_extras: dict = field(default_factory=dict)


def parse_matrix(path: Path) -> MatrixDoc:
    text = Path(path).read_text()
    if not text.startswith("---\n"):
        raise MatrixDocError(f"{path}: missing YAML front-matter")
    _, frontmatter, body = text.split("---\n", 2)
    data = yaml.safe_load(frontmatter)
    if not isinstance(data, dict):
        raise MatrixDocError(f"{path}: front-matter must be a YAML mapping")
    for field in _TOP:
        if field not in data:
            raise MatrixDocError(f"{path}: missing required field '{field}'")
    for key in ("matrix", "batch"):
        if not isinstance(data[key], dict):
            raise MatrixDocError(f"{path}: '{key}' must be a YAML mapping, got {type(data[key]).__name__}")
    for field in _MATRIX:
        if field not in data["matrix"]:
            raise MatrixDocError(f"{path}: missing matrix.{field}")
    for field in _BATCH:
        if field not in data["batch"]:
            raise MatrixDocError(f"{path}: missing batch.{field}")
    if _PLACEHOLDER.search(frontmatter) or _PLACEHOLDER.search(body):
        raise MatrixDocError(f"{path}: contains placeholder (TBD/TODO/<FILL>)")
    m, b = data["matrix"], data["batch"]
    # Strict, no coercion: this is a pre-registration parser. int(2.7) would
    # silently truncate the replicate count and bool is an int subclass
    # (int(True)==1), so both must be rejected outright. A quoted YAML "5"
    # (str) is rejected too — strictness is correct for a sealed design doc.
    repeats = m.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise MatrixDocError(f"{path}: matrix.repeats must be an integer, got {repeats!r}")
    if repeats < 1:
        raise MatrixDocError(f"{path}: matrix.repeats must be >= 1, got {repeats}")
    # Optional run_extras: an experiment-level run-config override carried to the
    # fleet. Absent -> {}. Must be a mapping with only allowlisted keys, so an
    # un-plumbed knob fails LOUDLY at parse instead of silently running incumbent.
    run_extras = data.get("run_extras", {})
    if run_extras is None:
        run_extras = {}
    if not isinstance(run_extras, dict):
        raise MatrixDocError(
            f"{path}: run_extras must be a YAML mapping, got {type(run_extras).__name__}"
        )
    unknown = [k for k in run_extras if k not in _RUN_EXTRAS_ALLOWED]
    if unknown:
        raise MatrixDocError(
            f"{path}: unknown run_extras key(s) {sorted(unknown)}; "
            f"allowed: {list(_RUN_EXTRAS_ALLOWED)}"
        )
    # Validate run_extras values loudly during pre-registration parsing. A typo'd smote_enabled (e.g. "ture") must NOT coerce to false and run
    # the whole registered arm as the incumbent; an un-plumbed variant/target must
    # fail here, not mid-run on the cloud. Lazy import so matrix_doc stays usable
    # without flowerfl. Reuses the same validators the runner/client enforce, so
    # doc-parse and run-time can't diverge. smote_enabled is normalized to a real
    # bool in the stored run_extras so the manifest carries a canonical value.
    if run_extras:
        run_extras = dict(run_extras)
        from flowerfl.smote_resampler import (
            coerce_smote_enabled, validate_smote_variant, normalize_smote_target,
        )
        try:
            if "smote_enabled" in run_extras:
                run_extras["smote_enabled"] = coerce_smote_enabled(run_extras["smote_enabled"])
            if "smote_variant" in run_extras:
                validate_smote_variant(run_extras["smote_variant"])
            if "smote_target" in run_extras:
                normalize_smote_target(run_extras["smote_target"])
            # Stage-F knobs (checklist item 7): same loud-at-parse contract —
            # the two bools reuse the strict smote_enabled coercer (typo'd
            # strings raise, stored value normalized to a real bool) and
            # weight_mode reuses the single-source validator the client
            # enforces, so doc-parse and run-time can't diverge.
            for bool_key in ("update_match", "smote_semantic_target",
                             "normalize_train_only"):
                if bool_key in run_extras:
                    try:
                        run_extras[bool_key] = coerce_smote_enabled(run_extras[bool_key])
                    except ValueError as e:
                        raise ValueError(f"{bool_key}: {e}") from e
            if "weight_mode" in run_extras:
                from flowerfl.resampling_manifest import validate_weight_mode
                validate_weight_mode(run_extras["weight_mode"])
            # Cross-field guard: semantic policy + a target
            # of exactly 1 parses per-field (legacy allows (0, 1]) but is
            # fatal in-run for semantic arms — fail at pre-registration.
            if run_extras.get("smote_semantic_target"):
                from flowerfl.smote_resampler import validate_semantic_target_combo
                validate_semantic_target_combo(run_extras.get("smote_target"))
            # H3 cohort declaration: validate against the SAME enum server_app
            # enforces (single source), and store the canonical lowercase value
            # so the manifest carries exactly what the runner will receive.
            if "fp_cohort" in run_extras:
                from flowerfl.fingerprint_registry import CalibrationCohort
                raw = str(run_extras["fp_cohort"] or "").strip().lower()
                try:
                    run_extras["fp_cohort"] = CalibrationCohort(raw).value
                except ValueError:
                    raise ValueError(
                        f"fp_cohort: unknown cohort {run_extras['fp_cohort']!r}; "
                        f"expected one of {[c.value for c in CalibrationCohort]}"
                    ) from None
            # H4 evaluation population (erratum-A E4): validate the closed
            # value set at pre-registration parse and store the canonical
            # lowercase value, so a typo'd manifest can never silently run the
            # legacy sampled holdout as a sealed-test unit (the H4 scorer
            # refuses units without eval_split=sealed_test custody).
            if "eval_split" in run_extras:
                raw = str(run_extras["eval_split"] or "").strip().lower()
                if raw not in _EVAL_SPLIT_VALUES:
                    raise ValueError(
                        f"eval_split: unknown value {run_extras['eval_split']!r}; "
                        f"expected one of {list(_EVAL_SPLIT_VALUES)}"
                    )
                run_extras["eval_split"] = raw
            # Erratum B § B1: strict-bool observe knob — a typo'd value must
            # NOT coerce and silently run the calibration fleet ENFORCING
            # (that would both distort the aggregate trajectory and starve
            # the honest-row pool the cuts are calibrated from). Reuses the
            # strict smote_enabled coercer; stored value normalized to a
            # real bool so the manifest carries a canonical value.
            if "h2p_observe_only" in run_extras:
                try:
                    run_extras["h2p_observe_only"] = coerce_smote_enabled(
                        run_extras["h2p_observe_only"]
                    )
                except ValueError as e:
                    raise ValueError(f"h2p_observe_only: {e}") from e
            # Erratum B § B2: closed cuts-version set, canonical lowercase —
            # explicit selection only, never auto-detect.
            if "h2p_cuts_version" in run_extras:
                raw = str(run_extras["h2p_cuts_version"] or "").strip().lower()
                if raw not in _H2P_CUTS_VERSIONS:
                    raise ValueError(
                        f"h2p_cuts_version: unknown value "
                        f"{run_extras['h2p_cuts_version']!r}; expected one of "
                        f"{list(_H2P_CUTS_VERSIONS)}"
                    )
                run_extras["h2p_cuts_version"] = raw
            # H3 registry candidate policy: same single-source discipline as the
            # cohort — validate against the enum the registry itself enforces.
            if "fp_registry_policy" in run_extras:
                from flowerfl.fingerprint_registry import RegistryPolicy
                raw = str(run_extras["fp_registry_policy"] or "").strip().lower()
                try:
                    run_extras["fp_registry_policy"] = RegistryPolicy(raw).value
                except ValueError:
                    raise ValueError(
                        "fp_registry_policy: unknown policy "
                        f"{run_extras['fp_registry_policy']!r}; expected one of "
                        f"{[p.value for p in RegistryPolicy]}"
                    ) from None
        except ValueError as e:
            raise MatrixDocError(f"{path}: invalid run_extras value — {e}") from e
    return MatrixDoc(
        exp_id=str(data["exp_id"]),
        slug=str(data["slug"]),
        hypothesis=str(data["hypothesis"]),
        methodology_version=str(data["methodology_version"]),
        defenses=list(m["defenses"]),
        scenarios=list(m["scenarios"]),
        seeds=[int(s) for s in m["seeds"]],
        mode=str(m["mode"]),
        max_per_client=int(m["max_per_client"]),
        rounds=int(m["rounds"]),
        repeats=repeats,
        job_queue=str(b["job_queue"]),
        job_definition=str(b["job_definition"]),
        body=body,
        path=Path(path),
        run_extras=dict(run_extras),
    )
