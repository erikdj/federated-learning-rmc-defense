"""Artifact access (S3 / local-dirs) and checks 1 + 6 for
scripts/audit_calibration_gate.py.

S3 mode reuses praxis_exp.storage / praxis_exp.manifest / praxis_exp.integrity
exactly as the fleet does (same ObjectStore dependency-injection contract);
local-dirs mode reads already-downloaded artifacts directly off disk.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Protocol

from praxis_exp import storage
from praxis_exp.storage import InMemoryObjectStore, ObjectStore
from praxis_exp.units import Unit

from _calibration_gate_lib import MODE_TO_OPTIMIZER_STATE, defense_token_for
from _calibration_gate_types import CheckResult, UnitRef

DEFAULT_MLFLOW_URI = "http://localhost:5001"


# ---------------------------------------------------------------------------
# GateStore: ObjectStore + LastModified
# ---------------------------------------------------------------------------


class GateStore(ObjectStore, Protocol):
    """praxis_exp.storage.ObjectStore's head()/size() intentionally don't
    surface a timestamp. S3's head_object() response DOES carry one
    ("LastModified"), but the shared interface doesn't expose it -- the gate
    needs write ORDER (done-marker committed last, integrity.py's contract),
    so this extends the interface here rather than editing the shared
    praxis_exp/storage.py module used by the whole fleet."""

    def last_modified(self, key: str) -> float | None: ...


class FakeGateStore(InMemoryObjectStore):
    """Test double for GateStore. Assigns a strictly increasing counter as a
    fake timestamp on every write, in call order -- praxis_exp.integrity.persist_unit
    (imported by tests, not reimplemented) writes result -> signal -> marker in
    that exact order, so building fixtures via persist_unit() produces a
    correct, deterministic ordering for free."""

    def __init__(self) -> None:
        super().__init__()
        self._counter = 0
        self._last_modified: dict[str, float] = {}

    def put_bytes(self, key: str, data: bytes, *, if_none_match: bool = False) -> None:
        super().put_bytes(key, data, if_none_match=if_none_match)
        self._touch(key)

    def put_file(self, key: str, path: Path) -> None:
        super().put_file(key, path)
        self._touch(key)

    def _touch(self, key: str) -> None:
        self._counter += 1
        self._last_modified[key] = float(self._counter)

    def last_modified(self, key: str) -> float | None:
        return self._last_modified.get(key)


class S3GateStore(storage.S3ObjectStore):
    """Production GateStore: praxis_exp.storage.S3ObjectStore + last_modified()
    via head_object()'s "LastModified" field. boto3 client is injected by the
    caller (main()); this class itself never constructs one."""

    _NOT_FOUND = {"NoSuchKey", "404"}  # mirrors praxis_exp/storage.py's _NOT_FOUND

    def last_modified(self, key: str) -> float | None:
        from botocore.exceptions import ClientError

        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in self._NOT_FOUND:
                return None
            raise
        return resp["LastModified"].timestamp()


# ---------------------------------------------------------------------------
# Unit construction
# ---------------------------------------------------------------------------


def unit_ref_from_manifest_unit(unit: Unit) -> UnitRef:
    expected_state = MODE_TO_OPTIMIZER_STATE.get(unit.mode, "reset")
    return UnitRef(
        unit_id=unit.unit_id,
        config=unit.config,
        scenario=unit.scenario,
        seed=unit.seed,
        expected_optimizer_state=expected_state,
        defense_token=defense_token_for(unit.config),
        rounds=unit.rounds,  # manifest-declared rounds (praxis_exp.units.Unit)
    )


class ExpectedUnitsError(RuntimeError):
    """Raised when the expected unit set cannot be resolved (bad manifest /
    missing design doc). Local-dirs mode REFUSES to fall back to globbing
    present result files -- discovery-by-glob let the gate PASS on incomplete
    downloads because absent/malformed units never produced a UnitRef."""


def load_local_manifest(manifest_path: Path, expected_exp_id: str) -> tuple[dict, list[Unit]]:
    """Read a downloaded manifest.json: (meta, units). Same payload shape and
    schema checks as praxis_exp.manifest.read_manifest (which requires an
    ObjectStore, hence this local-file twin). Unknown meta keys (e.g. 's
    upcoming prior_launches / launched_at) are tolerated by construction --
    only known keys are ever read.

    The manifest's exp_id must MATCH the audited experiment: unit_ids do not embed the exp id, so trusting a
    same-matrix manifest from another sweep verbatim would audit that sweep's
    artifacts and could emit a passing report for the wrong experiment."""
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpectedUnitsError(f"manifest {manifest_path} unreadable: {exc}") from exc
    for key in ("exp_id", "meta", "units"):
        if key not in data:
            raise ExpectedUnitsError(f"manifest {manifest_path} missing top-level key {key!r}")
    if data["exp_id"] != expected_exp_id:
        raise ExpectedUnitsError(
            f"manifest {manifest_path} is for exp_id {data['exp_id']!r}, but this audit "
            f"was invoked for {expected_exp_id!r} -- refusing to audit another sweep's "
            "artifacts (unit_ids do not embed the exp id, so a same-matrix manifest "
            "would silently produce a report for the wrong experiment)"
        )
    try:
        units = [Unit(**u) for u in data["units"]]
    except TypeError as exc:
        raise ExpectedUnitsError(
            f"manifest {manifest_path} has an incompatible unit schema: {exc}"
        ) from exc
    return data["meta"], units


def load_design_matrix_units(repo_root: Path, exp_id: str) -> list[Unit]:
    """Expected unit set from the experiment's design doc
    (docs/experiments/EXP-NNN-*.md matrix front-matter) -- the same source
    `praxis exp launch-matrix` expands, expanded with the same
    praxis_exp.units.expand_matrix (identical iteration order), so the audit
    expectation matches the launch expansion exactly.

    Deliberately does NOT reuse praxis_exp.matrix_doc.parse_matrix verbatim:
    its TBD-placeholder rejection is a LAUNCH guard (batch.job_queue /
    batch.job_definition legitimately read TBD before deploy_stack.sh) and
    the audit only needs the matrix block."""
    import yaml
    from praxis_exp.matrix_launch import MatrixLaunchError, _find_matrix_doc
    from praxis_exp.units import expand_matrix

    try:
        doc_path = _find_matrix_doc(Path(repo_root), exp_id)
    except MatrixLaunchError as exc:
        raise ExpectedUnitsError(
            f"no design doc for {exp_id}: {exc} -- pass --manifest-path instead"
        ) from exc
    text = doc_path.read_text()
    if not text.startswith("---\n"):
        raise ExpectedUnitsError(f"{doc_path}: missing YAML front-matter")
    _, frontmatter, _ = text.split("---\n", 2)
    data = yaml.safe_load(frontmatter)
    matrix = data.get("matrix") if isinstance(data, dict) else None
    if not isinstance(matrix, dict):
        raise ExpectedUnitsError(f"{doc_path}: no 'matrix' mapping in front-matter")
    for field in ("defenses", "scenarios", "seeds", "mode", "max_per_client", "rounds"):
        if field not in matrix:
            raise ExpectedUnitsError(f"{doc_path}: missing matrix.{field}")
    return expand_matrix(
        list(matrix["defenses"]), list(matrix["scenarios"]),
        [int(s) for s in matrix["seeds"]], str(matrix["mode"]),
        int(matrix["max_per_client"]), int(matrix["rounds"]),
    )


# ---------------------------------------------------------------------------
# Check 1: S3 integrity (result + signal + done-marker, marker written last)
# ---------------------------------------------------------------------------


def check_s3_integrity(store: GateStore, exp_id: str, unit: UnitRef) -> CheckResult:
    rk = storage.result_key(exp_id, unit.unit_id)
    sk = storage.signal_key(exp_id, unit.unit_id)
    mk = storage.marker_key(exp_id, unit.unit_id)
    problems: list[str] = []
    if not store.head(rk):
        problems.append("result JSON missing in S3")
    if not store.head(sk):
        problems.append("signal log missing in S3")
    if not store.head(mk):
        problems.append("done-marker missing in S3")
    if not problems:
        r_lm, s_lm, m_lm = store.last_modified(rk), store.last_modified(sk), store.last_modified(mk)
        if None in (r_lm, s_lm, m_lm):
            problems.append("last_modified unavailable for one or more objects")
        elif not (m_lm >= r_lm and m_lm >= s_lm):
            problems.append(
                f"done-marker LastModified ({m_lm}) is not >= result ({r_lm}) / "
                f"signal ({s_lm}) -- commit-order violation (integrity.py's contract)"
            )
    return CheckResult(
        "s3_integrity", unit.unit_id, not problems, True,
        "; ".join(problems) if problems else "result+signal+marker present, marker written last",
    )


def check_local_presence(unit: UnitRef, result_path: Path, signal_path: Path) -> CheckResult:
    """Reduced local-dirs equivalent of check_s3_integrity: no done-marker or
    LastModified exists offline, so this only confirms both artifacts are on
    disk. Intentionally a distinct check name from s3_integrity so a report
    never claims S3 durability it did not verify."""
    problems: list[str] = []
    if not result_path.is_file():
        problems.append(f"result JSON missing: {result_path}")
    if not signal_path.is_file():
        problems.append(f"signal log missing: {signal_path}")
    return CheckResult(
        "local_presence", unit.unit_id, not problems, True,
        "; ".join(problems) if problems else
        "result + signal present on disk (local-dirs mode: no done-marker/"
        "LastModified check available offline)",
    )


def materialize_s3_artifacts(
    store: GateStore, exp_id: str, unit: UnitRef, tmpdir: Path
) -> tuple[Path | None, Path | None]:
    """Download the raw result JSON + signal log bytes to tmpdir so
    audit_run_instrumentation.audit() (which opens real file paths) can be
    reused unmodified. Deliberately does NOT parse: a truncated/corrupt S3
    object raising out of json.loads here aborted run_gate_s3 before the
    report rendered and left the remaining units unaudited -- parsing happens in the runner, where a
    failure becomes a per-unit required s3_artifact_parse CheckResult."""
    rk = storage.result_key(exp_id, unit.unit_id)
    sk = storage.signal_key(exp_id, unit.unit_id)
    result_path = signal_path = None
    if store.head(rk):
        result_path = tmpdir / f"{unit.unit_id}.json"
        result_path.write_bytes(store.get_bytes(rk))
    if store.head(sk):
        signal_path = tmpdir / f"{unit.unit_id}.jsonl"
        signal_path.write_bytes(store.get_bytes(sk))
    return result_path, signal_path


# ---------------------------------------------------------------------------
# Check 6: MLflow (S3-mode only) -- REST API via urllib, injectable fetcher
# ---------------------------------------------------------------------------


class MlflowGateError(RuntimeError):
    """Raised when the MLflow experiment for this gate cannot be resolved."""


def default_mlflow_fetcher(url: str, data: bytes | None = None) -> dict:
    """Real HTTP transport for the MLflow REST API. No new dependency: stdlib
    urllib only. Tests inject a fake fetcher instead of hitting the network."""
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 -- operator-supplied tracking URI, not user input
        return json.loads(resp.read())


def _tags_dict(run: dict) -> dict[str, str]:
    tags = (run.get("data") or {}).get("tags") or []
    return {t["key"]: t["value"] for t in tags}


def _params_dict(run: dict) -> dict[str, str]:
    params = (run.get("data") or {}).get("params") or []
    return {p["key"]: p["value"] for p in params}


def _input_value(
    params: dict[str, str], tags: dict[str, str],
    params_keys: tuple[str, ...], tags_keys: tuple[str, ...],
) -> str | None:
    """First present value across ``params_keys`` (preferred — the PR #15
    canonical location for experiment inputs) then ``tags_keys`` (legacy
    fallback so pre-enrichment runs, which carried inputs as tags, still audit)."""
    for k in params_keys:
        if params.get(k) is not None:
            return params[k]
    for k in tags_keys:
        if tags.get(k) is not None:
            return tags[k]
    return None


def _current_parent_run_id(runs: list[dict], exp_id: str) -> str | None:
    """The current launch's parent run id among fetched runs: the newest
    non-FAILED run tagged ``exp_id=<exp_id>`` (only parent runs carry that tag;
    aborted launches are terminated FAILED). Used to scope the run-presence
    check to the launch being audited."""
    parents = []
    for run in runs:
        if _tags_dict(run).get("exp_id") == exp_id:
            info = run.get("info") or {}
            parents.append((
                info.get("start_time") or 0,
                info.get("status"),
                info.get("run_id") or info.get("run_uuid"),
            ))
    parents.sort(key=lambda p: p[0], reverse=True)  # newest launch first
    for _start, status, rid in parents:
        if status != "FAILED":
            return rid
    return None


def fetch_experiment_runs(
    mlflow_uri: str, experiment_name: str, fetcher: Callable[[str, bytes | None], dict]
) -> list[dict]:
    exp_url = (
        f"{mlflow_uri}/api/2.0/mlflow/experiments/get-by-name"
        f"?experiment_name={urllib.parse.quote(experiment_name)}"
    )
    exp_resp = fetcher(exp_url, None)
    experiment = exp_resp.get("experiment")
    if not experiment:
        raise MlflowGateError(f"MLflow experiment {experiment_name!r} not found at {mlflow_uri}")
    body = json.dumps({
        "experiment_ids": [experiment["experiment_id"]], "max_results": 1000,
    }).encode()
    search_resp = fetcher(f"{mlflow_uri}/api/2.0/mlflow/runs/search", body)
    return search_resp.get("runs", [])


def check_mlflow_runs(
    exp_id: str,
    units: list[UnitRef],
    meta: dict,
    mlflow_uri: str,
    fetcher: Callable[[str, bytes | None], dict],
) -> list[CheckResult]:
    experiment_name = f"{exp_id}__calibration"
    try:
        runs = fetch_experiment_runs(mlflow_uri, experiment_name, fetcher)
    except MlflowGateError as exc:
        return [CheckResult("mlflow.experiment", None, False, True, str(exc))]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        # transport/parse failures (dead
        # localhost:5001 tunnel -> urllib.error.URLError which subclasses
        # OSError; socket timeout -> OSError; non-JSON 502 page ->
        # json.JSONDecodeError, a ValueError subclass) must become REQUIRED
        # per-unit failures, NOT an exception that aborts the CLI before the
        # report is rendered/written. One failure per unit so every unit's
        # report row shows its MLflow verification is unproven.
        return [
            CheckResult(
                "mlflow.fetch", unit.unit_id, False, True,
                f"MLflow fetch failed at {mlflow_uri}: "
                f"{type(exc).__name__}: {exc} -- run verification unproven",
            )
            for unit in units
        ]

    # Scope to the CURRENT launch's children before grouping. A design-family
    # experiment holds children from every launch (aborted + current) with the
    # SAME deterministic unit_ids, so an aborted launch's stale done-run could
    # false-pass, or two completed launches read as ambiguous. Filter to the
    # current non-FAILED parent's children when resolvable; if not (older runs
    # predate the exp_id/parentRunId tags) fall back to unscoped; see
    # _calibration_gate_store.py.
    parent_run_id = _current_parent_run_id(runs, exp_id)
    if parent_run_id is not None:
        runs = [r for r in runs if _tags_dict(r).get("mlflow.parentRunId") == parent_run_id]

    by_unit_id: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        uid = _tags_dict(run).get("unit_id")
        if uid:
            by_unit_id[uid].append(run)

    image_digest = meta.get("image_digest")
    results: list[CheckResult] = []
    for unit in units:
        matches = by_unit_id.get(unit.unit_id, [])
        # Count only the unit_status=done run. A unit_id can legitimately map to
        # MORE than one run: a reclaimed attempt retried under the same launch
        # (the mis-sealed/reconciled zombie + the completing retry), and the
        # children of aborted launches that never completed. Requiring exactly
        # one *run* is therefore wrong; the single successful/enriched run is the
        # one tagged unit_status=done — the others are FAILED/reconciled/killed

        done = [r for r in matches if _tags_dict(r).get("unit_status") == "done"]
        if len(done) != 1:
            results.append(CheckResult(
                "mlflow.run_presence", unit.unit_id, False, True,
                f"expected exactly 1 unit_status=done MLflow run for "
                f"unit_id={unit.unit_id!r}, found {len(done)} done of {len(matches)} total",
            ))
            continue
        tags = _tags_dict(done[0])
        params = _params_dict(done[0])
        # PR #15 param/tag isolation: experiment INPUTS live in PARAMS now
        # (config label -> PARAM 'defense', plus scenario/seed); only metadata
        # (unit_id, image_digest, unit_status, defense_token, ...) stays in
        # TAGS. Read inputs from params with a legacy tag/param fallback so
        # pre-enrichment runs (inputs as tags, 'config' param) still audit.
        # image_digest/unit_status remain tags. methodology_version/git_sha are
        # parent-run-only, so not checked per-unit here.
        problems = []
        for label, observed, expected in (
            ("config", _input_value(params, tags, ("defense", "config"), ("config",)), unit.config),
            ("scenario", _input_value(params, tags, ("scenario",), ("scenario",)), unit.scenario),
            ("seed", _input_value(params, tags, ("seed",), ("seed",)), str(unit.seed)),
        ):
            if observed != expected:
                problems.append(f"{label}: expected {expected!r}, got {observed!r}")
        if tags.get("image_digest") != image_digest:
            problems.append(f"tag image_digest: expected {image_digest!r}, got {tags.get('image_digest')!r}")
        if tags.get("unit_status") != "done":
            problems.append(f"tag unit_status: expected 'done', got {tags.get('unit_status')!r}")
        results.append(CheckResult(
            "mlflow.tags", unit.unit_id, not problems, True,
            "; ".join(problems) if problems else "run present, inputs (params) + unit_status=done match",
        ))
    return results
