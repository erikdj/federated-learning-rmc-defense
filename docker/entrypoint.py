"""AWS Batch array-job entrypoint: resolve this unit, run it, persist results.

Pure helpers (resolve_unit, should_skip, build_unit_tags, build_unit_params,
s3_uris_for_unit, trajectory_metrics, final_metrics, cold_start_model_tag,
build_unit_note) are unit-tested; main() does the I/O and is integration-tested
by the calibration runs.

MLflow enrichment (redesign spec 2026-07-12-mlflow-enrichment-redesign-design):
  - req 1 (lifecycle): _RunController owns the run's terminal status so it
    ALWAYS agrees with unit_status — runner failure ends FAILED, uncaught
    exception ends FAILED, Spot-reclaim SIGTERM ends KILLED (+ reclaim_reason).
  - req 2 (traces): RETIRED (GWU-47) — an FL training run has no call tree, so
    the per-round timeline is an mlflow.log_table, not a trace. resolve_unit and
    the skip-check still run before any MLflow contact so an already-committed
    unit exits 0 even with MLflow down (PR #13, comment 3567094768).
  - req 2/3 (params ≠ tags): experiment INPUTS -> PARAMS (build_unit_params);
    identity/provenance/status/links -> TAGS (build_unit_tags + start metadata).
    A key lives in exactly one of the two — no duplication.
  - req 3 (deterministic at START): _set_start_metadata sets params, tags, S3
    link tags, the CloudWatch log URL and the native dataset input at run
    creation, so a RUNNING/FAILED run is already fully navigable.
  - req 4 (native dataset): log_dataset_input logs the EdgeIIoT dataset via
    mlflow.log_input (source = the S3 dataset URI), plus metadata/features JSON
    as artifacts.
  - req 5 (system metrics): _enable_system_metrics (psutil-backed).
  - req 6 (S3 + CloudWatch links): s3_* tags + note.content + cloudwatch_log_url.
  - req 7 (final metrics): final_metrics adds wall_clock_sec + rounds_completed.
  - req 8 (native model): log_native_model logs a pytorch flavor + registry
    entry (best-effort; needs the MLflow 3.x server).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from praxis_exp.integrity import is_done, persist_unit
from praxis_exp.manifest import read_manifest, unit_for_index
from praxis_exp.mlflow_enrichment import (
    cloudwatch_log_url,
    dataset_digest_from_metadata,
    dataset_source_uri,
    unit_s3_tags,
)
from praxis_exp.runner_paths import model_filename, result_filename, signal_filename
from praxis_exp.storage import ObjectStore, S3ObjectStore
from praxis_exp.units import Unit

# config-label -> signal-log defense token
# (strategy_name.replace("Scenario","").lower()).
#
# VERIFIED against flowerfl/server_app.py line 76:
#   defense = strategy_name.replace("Scenario", "").lower() or "unknown"
# and run_phase4_flower.py build_strategy_for_config() lines 233-252:
#
#   "Krum"     -> ScenarioKrum.__name__       -> "krum"
#   "TrustScore"-> ScenarioTrustScore.__name__ -> "trustscore"
#   "Krum+TGE" -> ScenarioKrumTGE.__name__    -> "krumtge"
#   "TGE"      -> ScenarioTGEnsemble.__name__  -> "tgensemble"
#
# NOTE: The AWS Phase A plan guessed "FedAvg+TGE" -> "fedavgtge", but
# "FedAvg+TGE" is NOT in SUPPORTED_CONFIGS. The TGE-only config label is
# "TGE" (maps to ScenarioTGEnsemble -> "tgensemble"). Extend this dict if
# new configs are added to SUPPORTED_CONFIGS.
_DEFENSE_TOKEN: dict[str, str] = {
    "Krum":       "krum",
    "TrustScore": "trustscore",
    "Krum+TGE":   "krumtge",
    "TGE":        "tgensemble",
    "Krum+TGEprime": "krumtgeprime",  # ScenarioKrumTGEPrime lowered (server_app.py:78)
    "TGEprime":   "tgeprime",         # ScenarioTGEPrime lowered — TGE′, GWU-53
    "Krum+CS":    "krumcs",
    "TrustScore+CS": "trustscorecs",
    "FedMedian":  "fedmedian",
    "FedTrimmedAvg": "fedtrimmedavg",
    # H3 fingerprint arms (v1.10 § 5.1 "Required H3 path" names both labels).
    # Same convention: ScenarioTGEFP -> "tgefp", ScenarioKrumTGEFP ->
    # "krumtgefp". These two were the exact omission the 2026-08-08 dry-run
    # flagged as the EXP-016 attempt-1 failure class (§ 2.1 / F2).
    "TGE+FP":     "tgefp",            # ScenarioTGEFP lowered
    "Krum+TGE+FP": "krumtgefp",       # ScenarioKrumTGEFP lowered
    # H4 composition arms (spec 2026-08-16 § 2 + erratum-A; BUILD_CONTRACT).
    # Same class-name-lowered convention; unit-id tokens (h2p_fp_krum, ...)
    # come from the config label via replace('+','_').lower() as always.
    "H2P+FP+Krum": "h2pfpkrum",       # ScenarioH2PFPKrum lowered
    "H2P+FP":     "h2pfp",            # ScenarioH2PFP lowered
    "H2P+Krum":   "h2pkrum",          # ScenarioH2PKrum lowered
    "H2P+FP+TS":  "h2pfpts",          # ScenarioH2PFPTS lowered
    "H2P+TS":     "h2pts",            # ScenarioH2PTS lowered (erratum-A arm 9)
    "FedAvg":     "none",             # ScenarioNone lowered (undefended floor)
}

_DATASET_NAME = "edge_full_20_rmc"
_DATASET_DIR = "edge_full_20"

# val split matches launch.py's single-experiment convention; the test split is
# sealed until H4 (v1.3). Single source so runner_argv and the run PARAM agree.
_REPORTING_SPLIT = "val"


def defense_token(config: str) -> str:
    """Signal-log defense token for a config label. Raises if config is unmapped
    (guards against _DEFENSE_TOKEN drifting from SUPPORTED_CONFIGS)."""
    token = _DEFENSE_TOKEN.get(config)
    if token is None:
        raise KeyError(f"no signal-log defense token for config {config!r}; "
                       f"known: {sorted(_DEFENSE_TOKEN)}")
    return token


def resolve_unit(store: ObjectStore, exp_id: str, array_index: int) -> Unit:
    _, _, units = read_manifest(store, exp_id)
    return unit_for_index(units, array_index)


def resolve_manifest_run_extras(store: ObjectStore, exp_id: str) -> dict:
    """The manifest's experiment-level ``run_extras`` (GWU-59), or {} if absent.

    Read on the RUN path only (after the skip-check), so a done-marker unit still
    exits without any extra work. Both the original launch and an in-namespace
    refill resolve the SAME manifest by exp_id, so a refill reproduces the arm's
    run_extras automatically."""
    _, meta, _ = read_manifest(store, exp_id)
    return meta.get("run_extras", {}) or {}


def should_skip(store: ObjectStore, exp_id: str, unit: Unit) -> bool:
    return is_done(store, exp_id, unit.unit_id)


def build_unit_tags(
    unit: Unit, *, methodology_version: str, image_digest: str,
    dataset_name: str = _DATASET_NAME,
) -> dict[str, str]:
    """Child-run METADATA tags (redesign § 2): identity / provenance / dataset
    name only. The experiment INPUTS (defense/scenario/seed/mode/rounds/
    max_per_client/reporting_split) live in ``build_unit_params`` — a key
    appears in EXACTLY ONE of params/tags, never both. ``defense_token`` (the
    signal-log token, e.g. ``krum``) is kept here as searchable metadata,
    distinct from the ``defense`` PARAM (the config label, e.g. ``Krum``)."""
    return {
        "unit_id": unit.unit_id,
        "defense_token": defense_token(unit.config),
        "methodology_version": methodology_version,
        "image_digest": image_digest,
        "dataset_name": dataset_name,
    }


def build_unit_params(unit: Unit, *, reporting_split: str = _REPORTING_SPLIT) -> dict[str, str]:
    """Child-run PARAMS (redesign § 2): the experiment INPUTS this unit ran —
    disjoint from ``build_unit_tags`` (metadata). ``defense`` is the config
    label (``Krum``); the signal-log token lives in the ``defense_token`` tag."""
    return {
        "defense": unit.config,
        "scenario": unit.scenario,
        "seed": str(unit.seed),
        "mode": unit.mode,
        "rounds": str(unit.rounds),
        "max_per_client": str(unit.max_per_client),
        "reporting_split": reporting_split,
    }


def s3_uris_for_unit(bucket: str, exp_id: str, unit_id: str) -> dict[str, str]:
    """S3 link tags for this unit's result/signal/done artifacts (req 5).

    Delegates to praxis_exp.mlflow_enrichment.unit_s3_tags, itself built
    from the SAME praxis_exp.storage key functions persist_unit uses — no
    duplicated string logic.
    """
    return unit_s3_tags(bucket, exp_id, unit_id)


def trajectory_metrics(result: dict) -> list[tuple[str, float, int]]:
    """Per-round accuracy/f1/loss as (key, value, step) triples, step=server
    round. Ground-truthed against run_phase4_flower.py's
    parse_eval_trajectory() output schema: trajectory is a list of
    {"round": int, "f1": float, "accuracy": float, "loss": float} dicts.
    """
    points: list[tuple[str, float, int]] = []
    for entry in result.get("trajectory") or []:
        step = entry.get("round")
        if step is None:
            continue
        # precision/recall are present on new-image trajectories (PR #15) and
        # skipped on older ones — all five live-contract metrics when available.
        # The six per-class keys mirror the direct-log block in
        # run_phase4_flower.py so self-heal/backfill-replayed units get the same
        # MLflow series; absent on legacy logs => skipped.
        for key in ("accuracy", "precision", "recall", "f1", "loss",
                    "attack_precision", "attack_recall", "attack_f1",
                    "benign_precision", "benign_recall", "benign_f1"):
            value = entry.get(key)
            if value is not None:
                points.append((key, float(value), int(step)))
    return points


def final_metrics(result: dict) -> dict[str, float]:
    """Final summary metrics (redesign § 2, item 7), present-subset only.

    In addition to ``final_accuracy``/``final_f1``/``mean_accuracy`` (from the
    result JSON), derives ``final_loss`` (last trajectory row), ``wall_clock_sec``
    (``elapsed_seconds``) and ``rounds_completed`` (trajectory length) when the
    source fields are present — so a completed run reports how long it took and
    how far it got. Present-subset: a key is emitted only when derivable, so
    partial/failed results never fabricate zeros.

    Note: scripts/run_phase4_flower.py ALSO best-effort streams final_f1/
    final_accuracy + per-round metrics directly (keyed off PRAXIS_MLFLOW_RUN_ID,
    wrapped in its own try/except) — intentional defense-in-depth so the run's
    metrics land even if that inline streaming silently failed; this is the only
    place mean_accuracy / wall_clock_sec / rounds_completed get logged."""
    out: dict[str, float] = {}
    for key in ("final_accuracy", "final_f1", "mean_accuracy"):
        value = result.get(key)
        if value is not None:
            out[key] = float(value)
    trajectory = result.get("trajectory") or []
    if trajectory:
        last_loss = trajectory[-1].get("loss")
        if last_loss is not None:
            out["final_loss"] = float(last_loss)
        out["rounds_completed"] = float(len(trajectory))
    elapsed = result.get("elapsed_seconds")
    if elapsed is not None:
        out["wall_clock_sec"] = float(elapsed)
    return out


def unit_criteria_ok(result: dict) -> bool:
    """Per-unit ``criteria_ok`` tag value: a clean completion — the runner's A6
    condition (``return_code == 0`` AND a non-empty trajectory). Sweep-level
    design predictions are evaluated at the PARENT run by ``praxis exp ingest``;
    there are no per-unit pre-registered thresholds. Shared with the Lane D
    backfill (``praxis_exp/enrich.py``) so live and backfilled child runs tag
    identically."""
    return result.get("return_code") == 0 and bool(result.get("trajectory"))


def cold_start_model_tag(result: dict) -> Optional[str]:
    """The CS pkl path/name used by this unit, if any (req 6), read from the
    runner's own provenance block (scripts/run_phase4_flower.py already sets
    provenance["cs_model_path"] = CS_MODEL if use_cs else "") — no
    re-derivation of the CS_MODEL/use_cs routing logic here."""
    path = (result.get("provenance") or {}).get("cs_model_path")
    return path or None


def build_unit_note(unit: Unit, *, result: dict, console_url: str, result_filename: str) -> str:
    """Short markdown note (mlflow.note.content) for the child run: unit summary,
    RMC params, gate verdict, clickable S3 console link + result filename (req 5 /
    GWU-47 Lane C — set on BOTH the live and backfill paths). Well within the
    8000-char tag cap."""
    lines = [
        f"### {unit.unit_id}",
        "",
        f"- Config: `{unit.config}`  Scenario: `{unit.scenario}`  Seed: `{unit.seed}`",
        f"- Mode: `{unit.mode}`  Rounds: `{unit.rounds}`  Max/client: `{unit.max_per_client}`",
    ]
    if result.get("final_accuracy") is not None:
        lines.append(f"- final_accuracy: `{result['final_accuracy']}`")
    if result.get("final_f1") is not None:
        lines.append(f"- final_f1: `{result['final_f1']}`")
    lines.append(f"- criteria_ok: `{unit_criteria_ok(result)}`")
    lines += [
        f"- Result file: `{result_filename}`",
        f"- [S3 console]({console_url})",
    ]
    return "\n".join(lines)


def smote_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flags for the SMOTE knob, derived from manifest ``run_extras``.

    Returns [] unless run_extras enables SMOTE — so a sweep without run_extras
    (the incumbent) produces byte-identical argv. When enabled, appends
    ``--smote-enabled`` plus the variant/target flags (falling back to the runner
    defaults if the doc omitted them). Validation of the values themselves is
    handled downstream (run_phase4_flower.smote_provenance_fields + client_fn),
    which raise loudly on an unknown variant / invalid target.
    """
    if not run_extras:
        return []
    # Strict coercion: a hand-edited manifest with a typo'd
    # smote_enabled (e.g. "ture") must RAISE here, not silently coerce to False
    # and run the whole arm incumbent. Defense in depth — matrix_doc already
    # validates at pre-registration parse, but the manifest is hand-editable.
    from flowerfl.smote_resampler import coerce_smote_enabled
    if not coerce_smote_enabled(run_extras.get("smote_enabled", False)):
        return []
    argv = ["--smote-enabled"]
    if "smote_variant" in run_extras:
        argv += ["--smote-variant", str(run_extras["smote_variant"])]
    if "smote_target" in run_extras:
        argv += ["--smote-target", str(run_extras["smote_target"])]
    return argv


def stage_f_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flags for the Stage-F knobs, derived from manifest
    ``run_extras`` (DESIGN_STAGE_F §4/§5/§6; image-update checklist item 7).

    Returns [] when the keys are absent OR at their incumbent values, so a sweep
    without them produces byte-identical argv. Same strict-coercion contract as
    ``smote_argv_from_run_extras``: a hand-edited manifest with a typo'd bool
    (e.g. ``"ture"``) or weight mode (``"orginal"``) RAISES here — it must never
    silently run the arm on the incumbent defaults (the GWU-59 near-miss this
    chain exists to prevent).
    """
    if not run_extras:
        return []
    from flowerfl.smote_resampler import coerce_smote_enabled
    from flowerfl.resampling_manifest import validate_weight_mode

    argv: list[str] = []
    if "update_match" in run_extras:
        try:
            if coerce_smote_enabled(run_extras["update_match"]):
                argv += ["--update-match"]
        except ValueError as e:
            raise ValueError(f"update_match: {e}") from e
    if "weight_mode" in run_extras:
        mode = validate_weight_mode(run_extras["weight_mode"])
        if mode != "resampled":  # incumbent default appends nothing
            argv += ["--weight-mode", mode]
    if "smote_semantic_target" in run_extras:
        try:
            if coerce_smote_enabled(run_extras["smote_semantic_target"]):
                # Cross-field guard: a unit-valued target is
                # legal legacy but fatal semantic — reject before launch.
                from flowerfl.smote_resampler import validate_semantic_target_combo
                validate_semantic_target_combo(run_extras.get("smote_target"))
                argv += ["--smote-semantic-target"]
        except ValueError as e:
            raise ValueError(f"smote_semantic_target: {e}") from e
    return argv


def leakage_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flag for the m1 normalization-leak fix, derived from manifest
    ``run_extras`` (normalization audit finding).

    Returns [] when ``normalize_train_only`` is absent OR false (its incumbent
    value), so a sweep without it produces byte-identical argv and the canonical
    pre-registered pipeline is untouched. Same strict-coercion contract as
    ``smote_argv_from_run_extras``/``stage_f_argv_from_run_extras``: a hand-edited
    manifest with a typo'd bool (e.g. ``"ture"``, or a number) RAISES here rather
    than silently running the leak-on incumbent (the GWU-59 near-miss this chain
    exists to prevent).
    """
    if not run_extras or "normalize_train_only" not in run_extras:
        return []
    from flowerfl.smote_resampler import coerce_smote_enabled
    try:
        if coerce_smote_enabled(run_extras["normalize_train_only"]):
            return ["--normalize-train-only"]
    except ValueError as e:
        raise ValueError(f"normalize_train_only: {e}") from e
    return []


def fp_cohort_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flag declaring which LOCKED fingerprint τ/Σ cohort scores this
    run (v1.10 § 5.1 D9; post-τ-lock).

    Returns [] when ``fp_cohort`` is absent, so pre-lock runs and non-FP arms
    produce byte-identical argv. When present, validates against the SAME enum
    ``server_app`` enforces (lazy import, single source), so a typo'd cohort in
    a hand-edited manifest fails at the entrypoint rather than shipping a unit
    that the server refuses 4 hours into a Spot window. The declaration itself
    is mandatory post-lock for FP arms — the validation and adjudicating τ are
    different instruments, and server_app hard-refuses an undeclared choice.
    """
    if not run_extras or "fp_cohort" not in run_extras:
        return []
    from flowerfl.fingerprint_registry import CalibrationCohort
    raw = str(run_extras["fp_cohort"] or "").strip().lower()
    try:
        cohort = CalibrationCohort(raw)
    except ValueError:
        raise ValueError(
            f"fp_cohort: unknown cohort {run_extras['fp_cohort']!r}; "
            f"expected one of {[c.value for c in CalibrationCohort]}"
        ) from None
    return ["--fp-cohort", cohort.value]


def fp_registry_policy_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flag declaring WHICH CANDIDATE POOL a re-entry decision
    considers (methodology v1.49; corrected-instrument amendment § 1).

    Returns [] when ``fp_registry_policy`` is absent, so every pre-existing run
    produces byte-identical argv and gets the deployed flag-gated registry.
    When present, validates against the SAME enum the registry enforces (lazy
    import, single source), so a typo'd policy in a hand-edited manifest fails
    at the entrypoint rather than shipping a unit that silently runs the
    incumbent instrument the corrected H3 exists to replace.
    """
    if not run_extras or "fp_registry_policy" not in run_extras:
        return []
    from flowerfl.fingerprint_registry import RegistryPolicy
    raw = str(run_extras["fp_registry_policy"] or "").strip().lower()
    try:
        policy = RegistryPolicy(raw)
    except ValueError:
        raise ValueError(
            f"fp_registry_policy: unknown policy "
            f"{run_extras['fp_registry_policy']!r}; expected one of "
            f"{[p.value for p in RegistryPolicy]}"
        ) from None
    return ["--fp-registry-policy", policy.value]


def eval_split_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flag declaring the EVALUATION POPULATION (erratum-A E4).

    Returns [] when ``eval_split`` is absent, so every pre-H4 run produces
    byte-identical argv and keeps the legacy sampled holdout. When present,
    validates against the same closed set the runner parser enforces, so a
    typo'd manifest value fails at the entrypoint rather than shipping a unit
    that silently evaluated on the wrong population.
    """
    if not run_extras or "eval_split" not in run_extras:
        return []
    raw = str(run_extras["eval_split"] or "").strip().lower()
    if raw not in ("legacy", "sealed_test"):
        raise ValueError(
            f"eval_split: unknown value {run_extras['eval_split']!r}; "
            f"expected 'legacy' or 'sealed_test' (erratum-A E4)"
        )
    return ["--eval-split", raw]


def h2p_argv_from_run_extras(run_extras: "dict | None") -> list[str]:
    """Runner CLI flags for the erratum-B detector knobs (v1.53).

    Returns [] when neither key is present, so every pre-erratum run
    produces byte-identical argv (enforcing detector, v1 cuts). Validation
    mirrors the matrix_doc/runner discipline: a typo'd bool or an unknown
    cuts-version fails HERE, at the entrypoint, rather than shipping a unit
    that silently ran the wrong mode or the wrong cut table.
    """
    if not run_extras:
        return []
    argv: list[str] = []
    if "h2p_observe_only" in run_extras:
        raw = run_extras["h2p_observe_only"]
        if isinstance(raw, bool):
            enabled = raw
        else:
            token = str(raw).strip().lower()
            if token in ("1", "true", "yes", "on"):
                enabled = True
            elif token in ("0", "false", "no", "off", ""):
                enabled = False
            else:
                raise ValueError(
                    f"h2p_observe_only: unknown value {raw!r}; expected a "
                    f"boolean (erratum B refuses to guess between observing "
                    f"and enforcing)"
                )
        if enabled:
            argv += ["--h2p-observe-only"]
    if "h2p_cuts_version" in run_extras:
        raw = str(run_extras["h2p_cuts_version"] or "").strip().lower()
        if raw not in ("v1", "v2"):
            raise ValueError(
                f"h2p_cuts_version: unknown value "
                f"{run_extras['h2p_cuts_version']!r}; expected 'v1' or 'v2' "
                f"(erratum B § B2 — explicit selection, never auto-detect)"
            )
        argv += ["--h2p-cuts-version", raw]
    return argv


def runner_argv(
    unit: Unit, *, scenario_dir: str, out_dir: Path, run_extras: "dict | None" = None,
) -> list[str]:
    """argv for the per-unit runner subprocess.

    ``--rounds`` is passed explicitly (PR #13 P2, comment 3567046864): the
    runner defaults rounds to 50 (run_phase4_flower.py:1180), and this argv
    historically never carried it — a pre-existing omission (the pre-branch
    entrypoint lacked it too) that the enrichment surfaced by logging
    unit.rounds as the run param. Every unit field the manifest records must
    be what the subprocess actually executes, or the manifest/MLflow lie
    about the trajectory.

    ``run_extras`` (GWU-59) carries experiment-level run-config overrides from
    the manifest meta (e.g. SMOTE); absent/empty appends nothing, keeping the
    argv byte-identical to the incumbent so a non-SMOTE sweep is unaffected.
    """
    return [
        sys.executable, "scripts/run_phase4_flower.py",
        "--configs",   unit.config,
        "--modes",     unit.mode,
        "--seeds",     str(unit.seed),
        "--scenario",  f"{scenario_dir}/{unit.scenario}.json",
        "--max-per-client", str(unit.max_per_client),
        "--rounds",    str(unit.rounds),
        # single source with the reporting_split PARAM so the argv and MLflow can't drift.
        "--reporting-split", _REPORTING_SPLIT,
        "--output-dir", str(out_dir),
    ] + smote_argv_from_run_extras(run_extras) + stage_f_argv_from_run_extras(run_extras) \
      + leakage_argv_from_run_extras(run_extras) + fp_cohort_argv_from_run_extras(run_extras) \
      + fp_registry_policy_argv_from_run_extras(run_extras) \
      + eval_split_argv_from_run_extras(run_extras) \
      + h2p_argv_from_run_extras(run_extras)


def post_persist_enrichment(
    mlflow_mod: Any, unit: Unit, *, bucket: str, exp_id: str,
    result: dict, model_path: Path,
) -> bool:
    """Best-effort MLflow decoration AFTER persist_unit's done-marker commit
    .

    The S3 done-marker is the unit's source of truth. Once persist_unit
    returns, this unit HAS succeeded — a raise here would exit the container
    nonzero, and the Batch retry would then no-op on the done-marker,
    leaving a spurious "failed" attempt and inconsistent MLflow state. So
    the entire decoration block (S3 link tags, note, model artifact) is
    guarded: any failure prints a loud warning and returns normally.
    ``unit_status=done`` is attempted LAST, inside the guard, so only a
    fully decorated run carries it.

    ``mlflow_mod`` is the fluent mlflow module (set_tag/log_artifact) —
    injected so tests exercise this with a fake, per the module's
    pure-helper pattern.

    Returns ``True`` iff the ENTIRE guarded block completed, ``False`` if any
    step raised. The never-raise contract is unchanged — the
    return value only tells ``main()`` whether to set the ``live_enrichment=
    complete`` marker the finalizer's skip-complete fast path requires. A False
    return does NOT mean the unit failed: its done-marker is committed, so it is
    durably successful; only its MLflow decoration is incomplete (must be re-logged
    by the self-heal enrich, never skipped).
    """
    try:
        s3_tags = s3_uris_for_unit(bucket, exp_id, unit.unit_id)
        for k, v in s3_tags.items():
            mlflow_mod.set_tag(k, v)
        mlflow_mod.set_tag(
            "mlflow.note.content",
            build_unit_note(
                unit, result=result, console_url=s3_tags["s3_console_url"],
                result_filename=result_filename(unit),
            ),
        )
        if model_path.is_file():
            mlflow_mod.log_artifact(str(model_path))
            mlflow_mod.set_tag("model_file", model_path.name)
        # signal log as a dataset-by-source (context "signal") — no byte copy;
        # parity with the backfill path (GWU-47 Lane A). Deterministic S3 key.
        from praxis_exp.mlflow_client import build_signal_dataset
        mlflow_mod.log_input(
            build_signal_dataset(
                exp_id=exp_id, unit_id=unit.unit_id, bucket=bucket,
                defense_token=defense_token(unit.config),
            ),
            context="signal",
        )
        # criteria_ok on the live success path too — same logic as the backfill
        # so freshly-completed runs are not blind to a criteria_ok filter/gate
        # until a manual enrich pass.
        mlflow_mod.set_tag("criteria_ok", "true" if unit_criteria_ok(result) else "false")
        mlflow_mod.set_tag("unit_status", "done")
        return True
    except Exception as e:
        print(
            f"[mlflow-enrich] WARN: post-persist MLflow enrichment failed for "
            f"{unit.unit_id}: {e} — the unit's S3 done-marker is already "
            "committed (source of truth); treating the unit as successful."
        )
        return False


# ---------------------------------------------------------------------------
# req 1: run lifecycle — status must always agree with unit_status
# ---------------------------------------------------------------------------

class _RunController:
    """Owns the child run's terminal status so it can never disagree with
    ``unit_status`` (redesign § 1/§ 3). Replaces the old ``with
    mlflow.start_run()`` context manager, whose ``return`` on a runner failure
    sealed the run FINISHED. Every terminal path routes through ``terminate``,
    which is idempotent (first status wins) so a Spot-reclaim SIGTERM that
    marks the run KILLED is not later overwritten by the finally/except path.
    """

    def __init__(self, mlflow_mod: Any, run_id: str) -> None:
        self._mlflow = mlflow_mod
        self._run_id = run_id
        self._terminated = False
        self._committed = False
        self._committed_criteria_ok: Optional[bool] = None

    def terminate(self, status: str) -> None:
        if self._terminated:
            return
        self._terminated = True
        try:
            self._mlflow.end_run(status)
        except Exception as e:  # best-effort: never fail the unit on teardown
            print(f"[entrypoint] WARN: could not end run {self._run_id} as {status}: {e}")

    def mark_error(self) -> None:
        try:
            self._mlflow.set_tag("unit_status", "errored")
        except Exception as e:
            print(f"[entrypoint] WARN: could not tag unit_status=errored: {e}")

    def mark_committed(self, criteria_ok: Optional[bool] = None) -> None:
        """The unit's done-marker is written (persist_unit returned) — it has
        durably succeeded, and a Batch retry would skip it on the marker. So a
        SIGTERM arriving during the slower post-commit decoration must end the
        run FINISHED, not KILLED. ``criteria_ok``
        is captured so the SIGTERM path can still tag it if the signal lands
        before the final setter runs."""
        self._committed = True
        self._committed_criteria_ok = criteria_ok

    def _on_sigterm(self, signum: int, _frame: Any) -> None:
        """Spot reclaim / termination.

        If the unit has already committed its done-marker (``mark_committed``),
        it has durably SUCCEEDED — a retry would skip on the marker — so end the
        run FINISHED even though the signal arrived mid-decoration; flipping it
        to KILLED would make the run status lie about a committed unit.
        Otherwise mark KILLED + reclaim_reason and exit 143 (128+SIGTERM). On a
        hard host kill the status may not flush — the backfill (Lane D) reconciles.
        """
        if self._committed:
            print("[entrypoint] SIGTERM after commit; unit already durable — ending FINISHED")
            try:
                if self._committed_criteria_ok is not None:
                    self._mlflow.set_tag(
                        "criteria_ok", "true" if self._committed_criteria_ok else "false")
                self._mlflow.set_tag("unit_status", "done")
            except Exception as e:
                print(f"[entrypoint] WARN: could not tag criteria_ok/unit_status=done: {e}")
            self.terminate("FINISHED")
            raise SystemExit(0)
        print("[entrypoint] SIGTERM received (likely Spot reclaim); marking run KILLED")
        try:
            self._mlflow.set_tag("unit_status", "killed")
            self._mlflow.set_tag("reclaim_reason", f"spot_interruption (SIGTERM {signum})")
        except Exception as e:
            print(f"[entrypoint] WARN: could not tag reclaim: {e}")
        self.terminate("KILLED")
        raise SystemExit(143)

    def install_reclaim_handler(self) -> None:
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except Exception as e:  # signal only works on the main thread
            print(f"[entrypoint] WARN: could not install SIGTERM handler: {e}")


def _enable_system_metrics(mlflow_mod: Any) -> None:
    """req 5: enable psutil-backed CPU/mem/disk/net system-metrics logging.
    Best-effort — a missing psutil or older mlflow must not fail the unit."""
    try:
        mlflow_mod.system_metrics.enable_system_metrics_logging()
    except Exception as e:
        print(f"[mlflow-sysmetrics] WARN: could not enable system metrics: {e}")


def _read_dataset_metadata(dataset_dir: str) -> Optional[dict]:
    """The dataset's ``metadata.json`` (synced into the container), or None."""
    path = Path("data") / dataset_dir / "metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[mlflow-dataset] WARN: could not read {path}: {e}")
        return None


def log_dataset_input(
    mlflow_mod: Any, *, bucket: str, dataset_dir: str = _DATASET_DIR,
    dataset_name: str = _DATASET_NAME,
) -> bool:
    """req 4: log the native EdgeIIoT dataset as an MLflow input (source = the
    S3 dataset URI, context=training). Best-effort — build failures warn.

    Returns True iff the input logged, False from the guarded except. The never-raise contract is unchanged — the bool lets the caller
    (_set_start_metadata -> main's start_ok) gate the live_enrichment marker on the
    dataset input, which the self-heal backfill (_enrich_completed_unit) repairs."""
    try:
        from praxis_exp.mlflow_client import build_meta_dataset
        metadata = _read_dataset_metadata(dataset_dir)
        dataset = build_meta_dataset(
            name=dataset_name,
            source_uri=dataset_source_uri(bucket, dataset_dir),
            digest=dataset_digest_from_metadata(metadata),
        )
        mlflow_mod.log_input(dataset, context="training")
        return True
    except Exception as e:
        print(f"[mlflow-dataset] WARN: failed to log dataset input: {e}")
        return False


def _cloudwatch_url_from_env() -> str:
    """req 6: best-effort CloudWatch console link for this Batch unit's logs.
    Uses the log group ``/aws/batch/job`` and the stream if the environment
    exposes it (rare at run start); otherwise deep-links to the group."""
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    log_group = os.environ.get("PRAXIS_CW_LOG_GROUP", "/aws/batch/job")
    log_stream = (
        os.environ.get("AWS_BATCH_JOB_LOG_STREAM")
        or os.environ.get("PRAXIS_CW_LOG_STREAM")
        or None
    )
    return cloudwatch_log_url(region=region, log_group=log_group, log_stream=log_stream)


def _set_start_metadata(
    mlflow_mod: Any, unit: Unit, *, bucket: str, exp_id: str,
    methodology_version: str, image_digest: str, parent_run_id: Optional[str],
) -> bool:
    """req 3: set ALL deterministic metadata at run START (params, tags, S3
    link tags, CloudWatch URL, native dataset input, unit_status=running) so a
    RUNNING or FAILED run is already fully navigable — not only after success.

    Returns the native dataset-input outcome: True when the
    dataset input logged, False when it was swallowed. The params/tags/S3-link steps
    still RAISE on failure (main's start_ok except catches them); the dataset input
    is the one step whose own guard hides its failure, so its bool is surfaced here
    so main's start_ok witnesses it too (the backfill repairs the dataset input)."""
    if parent_run_id:
        mlflow_mod.set_tag("mlflow.parentRunId", parent_run_id)
    for k, v in build_unit_tags(
        unit, methodology_version=methodology_version, image_digest=image_digest,
    ).items():
        mlflow_mod.set_tag(k, v)
    for k, v in build_unit_params(unit).items():
        mlflow_mod.log_param(k, v)
    for k, v in s3_uris_for_unit(bucket, exp_id, unit.unit_id).items():
        mlflow_mod.set_tag(k, v)
    mlflow_mod.set_tag("cloudwatch_log_url", _cloudwatch_url_from_env())
    mlflow_mod.set_tag("unit_status", "running")
    return log_dataset_input(mlflow_mod, bucket=bucket)


def _model_num_features(dataset_dir: str) -> Optional[int]:
    """Input feature count for the model signature, from the dataset's
    ``metadata.json`` ``_meta.num_features`` (GWU-47 Lane D). ``None`` if absent
    — the caller then logs the model without a signature (still best-effort)."""
    meta = _read_dataset_metadata(dataset_dir)
    if meta and isinstance(meta.get("_meta"), dict):
        nf = meta["_meta"].get("num_features")
        if isinstance(nf, int) and nf > 0:
            return nf
    return None


def _pytorch_log_model(
    mlflow_mod: Any, model_obj: Any, *, registered_model_name: str,
    signature: Any = None, input_example: Any = None,
) -> Any:
    """Call mlflow.pytorch.log_model tolerating the 2.x (``artifact_path``) vs
    3.x (``name``) keyword rename. Returns the ``ModelInfo`` — its ``model_id``
    links final metrics to the logged model (GWU-47 Lane D)."""
    try:
        return mlflow_mod.pytorch.log_model(
            model_obj, name="model", registered_model_name=registered_model_name,
            signature=signature, input_example=input_example,
        )
    except TypeError:
        return mlflow_mod.pytorch.log_model(
            model_obj, artifact_path="model", registered_model_name=registered_model_name,
            signature=signature, input_example=input_example,
        )


def log_native_model(
    mlflow_mod: Any, *, model_path: Path, defense_token: str, model_dataset: str,
    result: Optional[dict] = None, _torch: Any = None, _create_model: Any = None,
) -> None:
    """req 8 / GWU-47 Lane D: log the final global model as a native ``pytorch``
    flavor, register it as ``praxis-{defense_token}``, attach a signature +
    deterministic ``input_example``, and link ``final_metrics(result)`` to the
    logged model via ``model_id``.

    Best-effort and fully guarded: the runner writes a state_dict (not a
    Module), so it is rehydrated via ``flowerfl.task.create_model`` before
    logging. Needs the MLflow 3.x server to actually persist (2.18 404s on
    ``log_model``) — a failure here only warns. ``_torch``/``_create_model``
    are injectable for unit tests."""
    try:
        if not Path(model_path).is_file():
            return
        torch = _torch
        if torch is None:
            import torch  # type: ignore
        loaded = torch.load(str(model_path), map_location="cpu")
        model_obj = loaded
        if isinstance(loaded, dict):  # a raw state_dict — rehydrate its module
            create_model = _create_model
            if create_model is None:
                from flowerfl.task import create_model  # type: ignore
            net = create_model(model_dataset)
            net.load_state_dict(loaded)
            net.eval()
            model_obj = net
        # Deterministic signature + input_example from the dataset feature count.
        # Use the RESOLVED torch (local var), not the raw _torch arg (None on the
        # real container path). Skip the signature if num_features is unavailable.
        signature = input_example = None
        num_features = _model_num_features(model_dataset)
        if num_features:
            import numpy as np
            from mlflow.models import infer_signature
            sample_x_np = np.zeros((1, num_features), dtype=np.float32)
            preds = model_obj(torch.from_numpy(sample_x_np)).detach().numpy()
            signature = infer_signature(sample_x_np, preds)
            input_example = sample_x_np
        info = _pytorch_log_model(
            mlflow_mod, model_obj, registered_model_name=f"praxis-{defense_token}",
            signature=signature, input_example=input_example,
        )
        # Link final metrics to the logged model version via model_id (3.x).
        model_id = getattr(info, "model_id", None)
        if model_id is not None and result is not None:
            for k, v in final_metrics(result).items():
                mlflow_mod.log_metric(k, v, model_id=model_id)
    except Exception as e:
        print(
            "[mlflow-model] WARN: native model logging failed (needs MLflow 3.x "
            f"server; best-effort): {e}"
        )


def _find_resumable_run_id(
    experiment_id: str, unit_id: str, parent_run_id: Optional[str],
    *, _client_factory: Any = None,
) -> Optional[str]:
    """GWU-45 Lane A: the newest prior MLflow run id for this ``(parent, unit)``,
    or ``None`` — so a Batch retry of an uncommitted unit RESUMES its run instead
    of minting a fresh child per attempt (root-cause zombie reduction, design
    § 5.1). Strictly best-effort: ANY failure returns ``None`` and the caller
    falls back to creating a new run — the resume lookup must NEVER fail the unit.

    Resume is only safe PARENT-SCOPED (L4): with no ``parent_run_id`` the
    ``tags.unit_id`` match spans the WHOLE experiment and could cross-resume a
    DIFFERENT launch's run for the same unit slug, so an unparented call returns
    ``None`` immediately WITHOUT searching. Production always injects
    ``PRAXIS_PARENT_RUN_ID`` (matrix_launch.py:367); a direct/manual submit that
    carries none must mint a FRESH run, never cross-resume.

    Deliberately NOT a ``PraxisMlflowClient`` method: that client builds
    ``Config()`` in ``__init__`` (mlflow_client.py), which raises ``ConfigError``
    without ``AWS_PROFILE`` — and the Batch container carries none. So this uses a
    bare, Config-free ``mlflow.tracking.MlflowClient`` and issues the SAME
    filter/order as ``PraxisMlflowClient.find_runs_by_unit`` (newest = last in
    start_time-ASC order). No status filter: the skip-check has already proved the
    unit uncommitted, so any prior ``(parent, unit)`` run is a superseded attempt
    safe to reuse. ``_client_factory`` seams the client for tests.

    ``import mlflow.tracking`` is LOCAL to this body on purpose: entrypoint.py has
    no module-level ``import mlflow`` (every mlflow import is function-local), so a
    module-level helper referencing ``mlflow.tracking.MlflowClient`` WITHOUT a
    local import would raise ``NameError`` on the default (``_client_factory=None``)
    path — which the best-effort ``except`` below would silently swallow into the
    create-new-run fallback, killing resume-by-unit in production while
    factory-injected tests still pass.
    """
    # L4: resume is only safe parent-scoped — no parent -> no resume, no search.
    if not parent_run_id:
        return None
    try:
        import mlflow.tracking

        client = (_client_factory or mlflow.tracking.MlflowClient)()
        filter_string = (
            f"tags.unit_id = '{unit_id}' "
            f"and tags.`mlflow.parentRunId` = '{parent_run_id}'"
        )
        runs = client.search_runs(
            [experiment_id],
            filter_string=filter_string,
            order_by=["attributes.start_time ASC"],
            max_results=1000,
        )
        return runs[-1].info.run_id if runs else None
    except Exception as e:
        print(
            f"[entrypoint] WARN: resume-by-unit lookup failed for {unit_id!r}: {e} "
            "— falling back to a fresh run"
        )
        return None


def main(_store: ObjectStore | None = None, _mlflow: Any | None = None) -> int:
    exp_id = os.environ["PRAXIS_EXP_ID"]
    bucket = os.environ["PRAXIS_BUCKET"]
    index = int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])
    if _store is None:
        # imported inside main() so the module imports in the test env (tests inject fakes).
        import boto3
        store: ObjectStore = S3ObjectStore(bucket, boto3.client("s3"))
    else:
        store = _store

    # PR #13 P2 (comment 3567094768): resolve + skip-check run BEFORE any
    # tracking-server contact — and before tracing setup, which is MLflow
    # machinery, so these two calls carry no spans. A unit whose done-marker
    # is already committed must exit 0 even with MLflow down; otherwise a
    # transient outage sends an already-durable unit into a pointless Batch
    # retry loop (the retry would then skip on the marker anyway).
    # mlflow.set_experiment is a raising REST call; the enforced ordering is the
    # strictest one: ZERO mlflow attribute access on the skip path.
    unit = resolve_unit(store, exp_id, index)
    if should_skip(store, exp_id, unit):
        print(f"SKIP {unit.unit_id} (done-marker present)")
        return 0

    if _mlflow is None:
        # imported inside main() so the module imports in the test env (tests inject fakes).
        import mlflow
    else:
        mlflow = _mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    # GWU-45 Lane A: capture the Experiment so its experiment_id can scope the
    # resume-by-unit lookup below (mlflow.set_experiment returns the Experiment).
    experiment = mlflow.set_experiment(os.environ["PRAXIS_MLFLOW_EXPERIMENT"])
    _enable_system_metrics(mlflow)  # req 5 (before the run so the collector spans it)

    methodology_version = os.environ.get("PRAXIS_METHODOLOGY_VERSION", "")
    out_dir = Path("results") / exp_id / unit.unit_id

    # GWU-45 Lane A: resume the (parent, unit) run on a Batch retry instead of
    # minting a fresh child per attempt — a reclaimed uncommitted unit collapses
    # all its retries into ONE run, so reclaims stop accumulating zombies. The
    # skip-check above already proved the unit uncommitted, so any prior
    # (parent, unit) run is a superseded attempt safe to reuse. Strictly
    # best-effort: on any lookup failure or no match, fall back to a new run.
    parent_run_id = os.environ.get("PRAXIS_PARENT_RUN_ID")
    existing_run_id = _find_resumable_run_id(
        experiment.experiment_id, unit.unit_id, parent_run_id,
    )

    # req 1: own the run's terminal status explicitly (no `with start_run()`
    # context manager) — a bare `return` inside it sealed the run FINISHED
    # regardless of unit_status. Every terminal path routes through the
    # controller so status ∈ {FINISHED, FAILED, KILLED} always agrees with
    # unit_status; a SIGTERM handler marks Spot-reclaimed runs KILLED.
    run = (
        mlflow.start_run(run_id=existing_run_id) if existing_run_id
        else mlflow.start_run(run_name=unit.unit_id)
    )
    controller = _RunController(mlflow, run.info.run_id)
    controller.install_reclaim_handler()
    try:
        # req 3: deterministic metadata at run START (best-effort — enrichment
        # must never fail the unit; the S3 done-marker is the source of truth).
        # start_ok feeds the live_enrichment=complete marker below:
        # it is the _set_start_metadata return (the native dataset-input outcome),
        # and False on any exception from its params/tags/S3-link steps.
        start_ok = True
        try:
            start_ok = _set_start_metadata(
                mlflow, unit, bucket=bucket, exp_id=exp_id,
                methodology_version=methodology_version,
                image_digest=os.environ.get("PRAXIS_IMAGE_DIGEST", ""),
                parent_run_id=parent_run_id,  # hoisted above (GWU-45 Lane A)
            )
        except Exception as e:
            start_ok = False
            print(f"[mlflow-enrich] WARN: start-metadata enrichment failed: {e}")

        # The runner subprocess inherits the full container env, so the
        # runtime-provenance vars reach run_phase4_flower.py's provenance block
        # without explicit forwarding here: PRAXIS_IMAGE_DIGEST +
        # PRAXIS_LAUNCH_COMMIT are launch-supplied (matrix_launch/matrix_refill),
        # PRAXIS_RUNNER_COMMIT is image-baked at build time (runtime provenance).
        # Only the run-scoped vars below are added per unit.
        env = dict(
            os.environ,
            PRAXIS_MLFLOW_RUN_ID=run.info.run_id,
            PRAXIS_RAY_CPUS=os.environ.get("PRAXIS_RAY_CPUS", "8"),
        )
        rc = subprocess.call(
            runner_argv(
                unit,
                scenario_dir=os.environ.get("PRAXIS_SCENARIO_DIR", "rmc/scenarios"),
                out_dir=out_dir,
                # GWU-59: append SMOTE (and any future run_extras) flags from the
                # manifest so a fleet unit actually runs the declared arm.
                run_extras=resolve_manifest_run_extras(store, exp_id),
            ),
            env=env,
        )
        if rc != 0:
            # req 1: end FAILED (not FINISHED) so status agrees with unit_status.
            mlflow.set_tag("unit_status", "runner_failed")
            controller.terminate("FAILED")
            return rc

        result_path = out_dir / result_filename(unit)
        signal_path = Path("signals") / signal_filename(
            unit, defense_token=defense_token(unit.config)
        )

        # metrics: read the result JSON and log final + per-round trajectory
        # metrics. Best-effort — the actual experimental record is the
        # JSON/signal-log on disk/S3, not these MLflow metrics. metrics_ok feeds
        # the live_enrichment=complete marker below: a failure here
        # AFTER final_f1 is logged would otherwise leave the marker+final_f1 skip
        # predicate falsely proving completeness while mean_accuracy/wall_clock_sec/
        # rounds_completed/trajectory points are missing. A json.loads failure -> False
        # too (everything downstream is degraded).
        result: dict = {}
        metrics_ok = True
        try:
            result = json.loads(result_path.read_text())
            for key, value in final_metrics(result).items():
                mlflow.log_metric(key, value)
            for key, value, step in trajectory_metrics(result):
                mlflow.log_metric(key, value, step=step)
            cs_model = cold_start_model_tag(result)
            if cs_model:
                mlflow.set_tag("cold_start_model", cs_model)
        except Exception as e:
            metrics_ok = False
            print(f"[mlflow-metrics] WARN: failed to log result metrics: {e}")

        # result.json artifact — the ONE artifact the self-heal backfill repairs
        # (enrich._enrich_completed_unit -> _log_artifacts uploads result.json). Log it
        # LIVE too so a skip-complete run matches a backfilled run (live==backfill
        # parity), and gate the marker on it: artifacts_ok is
        # False if the upload raised OR the expected file was missing so it never logged.
        artifacts_ok = True
        try:
            if result_path.is_file():
                mlflow.log_artifact(str(result_path))
            else:
                artifacts_ok = False
        except Exception as e:
            artifacts_ok = False
            print(f"[mlflow-artifact] WARN: failed to log {result_path}: {e}")

        # req 4: dataset artifacts, if synced into the container (they are —
        # docker/Dockerfile syncs data/edge_full_20/ from S3 at container start).
        # NOT gated into artifacts_ok: the backfill does NOT re-upload these req-4
        # dataset artifacts, so the marker must not witness them (over-gating would
        # only cost a redundant re-log of something the backfill can't repair anyway).
        for fname in ("metadata.json", "features.json"):
            fpath = Path("data") / _DATASET_DIR / fname
            if fpath.is_file():
                try:
                    mlflow.log_artifact(str(fpath))
                except Exception as e:
                    print(f"[mlflow-artifact] WARN: failed to log {fpath}: {e}")

        # persist INSIDE the run: an IntegrityError propagates to the except
        # below, which ends the run FAILED (Batch retries).
        persist_unit(store, exp_id, unit.unit_id, result_path, signal_path)

        # The done-marker is committed — the unit has durably succeeded. Flag it
        # so a SIGTERM during the slower best-effort decoration below ends the
        # run FINISHED, not KILLED.
        controller.mark_committed(criteria_ok=unit_criteria_ok(result))

        # AFTER the done-marker commit everything is best-effort decoration
        # (S3 links + note, model artifact, unit_status=done last) — see
        # post_persist_enrichment: it never raises (PR #13 P2, comment 3566978588).
        # post_ok is False if any decoration step failed -> gates the marker below.
        post_ok = post_persist_enrichment(
            mlflow, unit, bucket=bucket, exp_id=exp_id, result=result,
            model_path=out_dir / model_filename(unit),
        )
        # req 8 (Lane B, best-effort; needs 3.x server): native pytorch model + registry.
        log_native_model(
            mlflow, model_path=out_dir / model_filename(unit),
            defense_token=defense_token(unit.config),
            model_dataset=os.environ.get("PRAXIS_MODEL_DATASET", _DATASET_DIR),
            result=result,
        )

        # criteria_ok + unit_status=done are GUARANTEED for a committed unit in
        # their own final best-effort step: post_persist_enrichment sets them
        # last, but if an earlier decoration call in its guarded block raised,
        # those tags would be skipped, leaving the run FINISHED but missing
        # criteria_ok / still 'running' so a gate keyed on them rejects a
        # durably-succeeded unit.
        try:
            mlflow.set_tag("criteria_ok", "true" if unit_criteria_ok(result) else "false")
            mlflow.set_tag("unit_status", "done")
            # live_enrichment=complete: the finalizer/reaper skip-complete
            # fast path may skip re-logging a run ONLY when this marker proves it was fully
            # logged in-container. unit_status=done is NOT sufficient — it is guaranteed for
            # a committed unit even when decoration partially failed (above / PR #15) or a
            # SIGTERM landed mid-decoration (_on_sigterm, which deliberately does NOT set
            # this marker). PRINCIPLE: the marker must witness EXACTLY the set the backfill
            # (_enrich_completed_unit) would repair — params/tags/S3-links AND the native
            # dataset input (start_ok — the dataset-input outcome is now surfaced through
            # _set_start_metadata's return, not swallowed), result + per-round metrics
            # (metrics_ok), the result.json artifact
            # (artifacts_ok), and note/S3-links/signal-dataset/criteria_ok (post_ok). Over-
            # gating is safe (a redundant re-log); under-gating strands a partially-logged
            # run. NOT gated on log_native_model or the req-4 dataset artifacts: the backfill
            # repairs neither, so their outcome is irrelevant to the skip decision.
            if start_ok and metrics_ok and artifacts_ok and post_ok:
                mlflow.set_tag("live_enrichment", "complete")
        except Exception as e:
            print(f"[entrypoint] WARN: could not set criteria_ok/unit_status=done: {e}")

        controller.terminate("FINISHED")
        print(f"DONE {unit.unit_id}")
        return 0
    except SystemExit:
        # the SIGTERM handler already marked the run KILLED — do not override it.
        raise
    except BaseException:
        # any uncaught failure (incl. persist IntegrityError): status = FAILED.
        controller.mark_error()
        controller.terminate("FAILED")
        raise


if __name__ == "__main__":
    sys.exit(main())
