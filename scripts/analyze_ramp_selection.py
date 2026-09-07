"""Dev-gate tenure-ramp selection analysis (amendment v1.6 § 3) — FROZEN PRE-DATA.

Pre-registered decision rule for the TGE tenure-ramp parameter, committed as
executable code BEFORE any H2 dev-sweep run exists (v1.6 § 3.6). Executed once,
at the v1.3 dev gate, after the Szeląg gate and BEFORE threshold freeze / H1
training.

Method (v1.6 § 3.3): the tenure gate is a pure deterministic function of the
logged raw expert scores, so every candidate ramp is evaluated OFFLINE by
re-blending `tge_gbdt_score` / `tge_lstm_score` at the logged `tge_tenure` —
zero additional federated compute. Per candidate, a single 10%-FPR operating
threshold is derived from the pooled HONEST population ONLY (the v1.3 F3/F8
frozen-threshold convention, via h2_threshold_pipeline.select_threshold —
malicious labels play no part in cutoff selection), then recall is computed
per (scenario, seed) at that fixed threshold.

Decision rule (v1.6 § 3.4, frozen verbatim): retain the incumbent ramp=8
unless a challenger beats it on the primary metric in ALL dev seeds AND by
>= 2 pp mean AND keeps mean cold-start recall within 1 pp. Multiple
qualifiers: largest mean margin, then closest to 8. This is a design-parameter
selection with a pre-registered rule — NOT a hypothesis test; no significance
claim is made from it.

Limitation (v1.6 § 3.5): open-loop — logged trajectories were generated with
the deployed ramp making the live decisions. The § 4 contingency (50-run TGE
dev re-fill) closes the loop if the selection switches the ramp.

Usage:
    python scripts/analyze_ramp_selection.py \\
        --signals-dir signals/ --defense krumtge --mode flower_persistent \\
        --out results/<gate-date>/ramp_selection/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from h2_threshold_pipeline import select_threshold  # noqa: E402
from rmc.tg_ensemble import TenureGatedDecisionRule  # noqa: E402

# --- Pre-registered constants (v1.6 § 3; do not edit without an amendment) ---
RAMP_CANDIDATES: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10)
INCUMBENT_RAMP: int = 8
SWITCH_MARGIN_PP: float = 2.0        # § 3.4 (b)
GUARD_TOLERANCE_PP: float = 1.0      # § 3.4 (c)
PRIMARY_SCENARIOS: tuple[str, ...] = ("S2", "S3", "S4")
REGISTERED_SCENARIOS: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4")  # § 3.2 selective input
COLDSTART_K: int = 3                 # v1.3 F6
TARGET_FPR: float = 0.10
MIN_TENURE: int = 2                  # gate's fixed lower knee
DEV_SEEDS: tuple[int, ...] = (42, 137, 256, 314, 500)  # data/seeds.json dev set


class RampAnalysisError(RuntimeError):
    """Raised when the analysis inputs violate the v1.6 § 3 preconditions."""


_RULES: dict[int, TenureGatedDecisionRule] = {}


def _rule(ramp: int) -> TenureGatedDecisionRule:
    if ramp not in _RULES:
        _RULES[ramp] = TenureGatedDecisionRule(min_tenure=MIN_TENURE, ramp_rounds=ramp)
    return _RULES[ramp]


def is_tge_scored(row: dict) -> bool:
    """False for rows whose client was filtered by an upstream plugin (e.g.
    Krum in the composed Krum+TGE chain) before TGE ever scored it — all TGE
    fields are null (truthful cid-keyed join, methodology v1.17). Such rows
    are ramp-invariant (excluded under EVERY candidate identically), so they
    are excluded from the re-blend — but callers must count and report them
    (no silent caps)."""
    return not (row.get("tge_phase") is None and row.get("tge_score") is None)


def reblend_score(row: dict, ramp: int) -> float:
    """Counterfactual TGE score at `ramp` for one signal-log row.

    Active-phase rows are re-blended from the raw expert scores at the logged
    `tge_tenure` (what the gate actually consumed). All other phases are
    ramp-invariant by construction and pass the logged score through.
    """
    phase = row.get("tge_phase")
    if phase == "active":
        gbdt, lstm = row.get("tge_gbdt_score"), row.get("tge_lstm_score")
        if gbdt is None or lstm is None:
            raise ValueError(
                f"active-phase row missing raw expert score(s) "
                f"(gbdt={gbdt!r}, lstm={lstm!r}) — violates the v1.6 § 5.4 "
                f"instrumentation contract; halting rather than skipping"
            )
        return _rule(ramp).compute_score(float(gbdt), float(lstm), int(row["tge_tenure"]))
    score = row.get("tge_score")
    if score is None:
        raise RampAnalysisError(
            f"row in phase {phase!r} has no usable tge_score — refusing to skip silently"
        )
    return float(score)


def parse_signal_filename(name: str):
    """`<mode>__<scenario>__<defense>__seed<N>.jsonl` -> (mode, scenario, defense, seed)."""
    if not name.endswith(".jsonl"):
        return None
    parts = name[: -len(".jsonl")].split("__")
    if len(parts) != 4 or not parts[3].startswith("seed"):
        return None
    try:
        seed = int(parts[3][len("seed"):])
    except ValueError:
        return None
    return parts[0], parts[1], parts[2], seed


def scenario_key(stem: str) -> str:
    """'S4_full_mix' -> 'S4'."""
    return stem.split("_")[0]


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _recall_at(rows: list[dict], scores: list[float], threshold: float,
               coldstart_only: bool, required: bool = True) -> float | None:
    """Recall (flag = score < threshold) over a cell, optionally cold-start scoped.

    `required=True` (primary S2/S3/S4 cells) hard-fails on a cell with no
    malicious ground truth — that means the dev sweep is broken. Descriptive
    cells (`required=False`) return None instead: the § 3.4 verdict never
    depends on them, so an attack-free descriptive input (e.g. an honest-
    control log) is reported as n/a rather than crashing the gate. The
    shipped S0/S1 are attack-bearing controls (amendment v1.4), so with
    documented dev-sweep inputs every cell has positives either way.
    """
    mal, tp = 0, 0
    for row, score in zip(rows, scores):
        if coldstart_only and int(row["tenure"]) > COLDSTART_K:
            continue
        if row["malicious_gt"]:
            mal += 1
            if score < threshold:
                tp += 1
    if mal == 0:
        if required:
            scope = "cold-start" if coldstart_only else "all-rounds"
            raise RampAnalysisError(f"primary cell has no malicious rows in {scope} "
                                    f"scope — cannot compute recall; halting")
        return None
    return tp / mal


def decide(primary: dict, guard: dict, *, seeds,
           candidates=RAMP_CANDIDATES, incumbent=INCUMBENT_RAMP,
           margin_pp=SWITCH_MARGIN_PP, guard_pp=GUARD_TOLERANCE_PP) -> dict:
    """Apply the v1.6 § 3.4 decision rule.

    primary/guard: {seed: {ramp: metric}} (fractions in [0, 1]).
    """
    mean_p = {r: mean(primary[s][r] for s in seeds) for r in candidates}
    mean_g = {r: mean(guard[s][r] for s in seeds) for r in candidates}

    qualifying = []
    for r in candidates:
        if r == incumbent:
            continue
        sweep = all(primary[s][r] > primary[s][incumbent] for s in seeds)
        margin_ok = (mean_p[r] - mean_p[incumbent]) * 100.0 >= margin_pp
        guard_ok = (mean_g[incumbent] - mean_g[r]) * 100.0 <= guard_pp
        if sweep and margin_ok and guard_ok:
            qualifying.append(r)

    if qualifying:
        selected = max(qualifying, key=lambda r: (mean_p[r], -abs(r - incumbent)))
        switched = True
    else:
        selected, switched = incumbent, False

    return {
        "selected": selected,
        "switched": switched,
        "qualifying": qualifying,
        "incumbent": incumbent,
        "mean_primary": {str(r): mean_p[r] for r in candidates},
        "mean_guard": {str(r): mean_g[r] for r in candidates},
        "rule": (f"switch iff sweep(all seeds) AND mean margin >= {margin_pp} pp "
                 f"AND cold-start guard within {guard_pp} pp (v1.6 § 3.4)"),
    }


def run_analysis(signals_dir: Path, *, defense: str, mode: str,
                 seeds=DEV_SEEDS, scenarios=REGISTERED_SCENARIOS,
                 out_dir: Path | None = None) -> dict:
    """`scenarios` is the whitelisted scenario-key set (v1.6 § 3.2): files
    whose key falls outside it are ignored and counted — a stray smoke or
    sensitivity log in the shared signals/ directory must not enter the
    pooled honest threshold. Every (scenario × seed) cell
    in the set is REQUIRED, so the verdict is a pure function of the
    registered input. The gate execution uses the registered default; only
    tests narrow it (the CLI exposes no override)."""
    signals_dir = Path(signals_dir)
    seeds = list(seeds)

    cells: dict[tuple[str, int], list[dict]] = {}
    stems: dict[str, str] = {}
    n_not_scored = 0
    n_ignored = 0
    for path in sorted(signals_dir.glob("*.jsonl")):
        parsed = parse_signal_filename(path.name)
        if parsed is None:
            continue
        f_mode, stem, f_defense, seed = parsed
        if f_mode != mode or f_defense != defense or seed not in seeds:
            continue
        skey = scenario_key(stem)
        if skey not in scenarios:
            n_ignored += 1
            continue
        if skey in stems and stems[skey] != stem:
            raise RampAnalysisError(
                f"two scenario stems map to {skey}: {stems[skey]!r} vs {stem!r} — "
                f"refusing to merge cells silently"
            )
        stems[skey] = stem
        rows = _load_rows(path)
        scored = [r for r in rows if is_tge_scored(r)]
        n_not_scored += len(rows) - len(scored)
        # SignalLogger opens files in APPEND mode and a re-run of the same
        # unit writes the same filename, so a file can accumulate multiple
        # runs — which would overweight this cell in the pooled threshold
        # and per-cell recall without any loud failure.
        stamps = {r.get("run_started_at") for r in scored}
        if len(stamps) > 1:
            raise RampAnalysisError(
                f"{path.name}: {len(stamps)} distinct run_started_at values — the "
                f"file mixes rows from multiple runs (signal logs append across "
                f"re-runs). Regenerate the cell from the unit's fresh run."
            )
        keys = [(r.get("server_round"), r.get("logical_cid")) for r in scored]
        if len(set(keys)) != len(keys):
            raise RampAnalysisError(
                f"{path.name}: duplicate (server_round, logical_cid) rows — "
                f"appended or duplicated run detected; refusing to pool a mixed cell."
            )
        cells.setdefault((skey, seed), []).extend(scored)
    if n_ignored:
        print(f"[ramp-selection] ignored {n_ignored} signal file(s) outside the "
              f"registered scenario set {scenarios} (v1.6 § 3.2)")
    if n_not_scored:
        print(f"[ramp-selection] {n_not_scored} rows excluded from the re-blend: "
              f"clients filtered upstream of TGE (ramp-invariant; v1.17)")

    missing = [(skey, s) for skey in scenarios for s in seeds
               if (skey, s) not in cells]
    if missing:
        raise RampAnalysisError(
            f"missing registered cells for defense={defense!r} mode={mode!r}: {missing} — "
            f"the pooled threshold is only deterministic on the full registered input"
        )

    all_keys = sorted({skey for skey, _ in cells})
    curve: dict[str, dict] = {}
    primary: dict[int, dict[int, float]] = {s: {} for s in seeds}
    guard: dict[int, dict[int, float]] = {s: {} for s in seeds}

    for ramp in RAMP_CANDIDATES:
        rescored = {cell: [reblend_score(row, ramp) for row in rows]
                    for cell, rows in cells.items()}
        # v1.6 § 3.3.2: the cutoff comes from the HONEST population only
        # (h2_threshold_pipeline convention). Selecting it recall-maximally
        # over pooled labels would tune the threshold on the same malicious
        # rows the curve is scored on, and would abort on legitimate
        # zero-recall candidates.
        honest_scores = [score
                         for cell, rows in cells.items()
                         for row, score in zip(rows, rescored[cell])
                         if not row["malicious_gt"]]
        if not honest_scores:
            raise RampAnalysisError(f"ramp {ramp}: no honest rows in the dev logs — "
                                    f"cannot derive a 10%-FPR threshold")
        threshold = select_threshold(honest_scores, TARGET_FPR)
        honest_fpr = sum(1 for s in honest_scores if s < threshold) / len(honest_scores)

        recall_all: dict[tuple[str, int], float | None] = {}
        recall_cs: dict[tuple[str, int], float | None] = {}
        for cell, rows in cells.items():
            required = cell[0] in PRIMARY_SCENARIOS
            recall_all[cell] = _recall_at(rows, rescored[cell], threshold,
                                          coldstart_only=False, required=required)
            recall_cs[cell] = _recall_at(rows, rescored[cell], threshold,
                                         coldstart_only=True, required=required)

        for s in seeds:
            primary[s][ramp] = mean(recall_all[(k, s)] for k in PRIMARY_SCENARIOS)
            guard[s][ramp] = mean(recall_cs[(k, s)] for k in PRIMARY_SCENARIOS)

        curve[str(ramp)] = {
            "threshold": threshold,
            "honest_fpr_at_threshold": honest_fpr,
            "per_seed_primary": {str(s): primary[s][ramp] for s in seeds},
            "per_seed_guard": {str(s): guard[s][ramp] for s in seeds},
            "per_cell_recall_all": {f"{k}__seed{s}": recall_all[(k, s)]
                                    for (k, s) in sorted(recall_all)},
        }

    verdict = decide(primary, guard, seeds=seeds)
    result = {
        "curve": curve,
        "verdict": verdict,
        "meta": {
            "spec": "docs/superpowers/specs/2026-07-11-praxis-experimental-design-v1.6.md",
            "defense": defense, "mode": mode, "seeds": seeds,
            "scenarios_analyzed": all_keys, "scenario_stems": stems,
            "primary_scenarios": list(PRIMARY_SCENARIOS),
            "registered_scenarios": list(scenarios),
            "coldstart_k": COLDSTART_K, "target_fpr": TARGET_FPR,
            "n_rows_not_tge_scored": n_not_scored,
            "n_files_ignored_unregistered": n_ignored,
        },
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
        (out_dir / "curve.json").write_text(json.dumps(result, indent=2))
        (out_dir / "curve.md").write_text(_curve_markdown(result))
    return result


def _curve_markdown(result: dict) -> str:
    verdict, curve = result["verdict"], result["curve"]
    lines = [
        "# Tenure-ramp selection (amendment v1.6 § 3)", "",
        f"**Selected: ramp = {verdict['selected']}** "
        f"({'SWITCHED from' if verdict['switched'] else 'incumbent retained,'} "
        f"{verdict['incumbent']})", "",
        "| ramp | mean primary (all-rounds recall) | mean guard (cold-start recall) | 10%-FPR threshold |",
        "|---|---|---|---|",
    ]
    for r in RAMP_CANDIDATES:
        lines.append(
            f"| {r}{' (incumbent)' if r == verdict['incumbent'] else ''} "
            f"| {verdict['mean_primary'][str(r)]:.4f} "
            f"| {verdict['mean_guard'][str(r)]:.4f} "
            f"| {curve[str(r)]['threshold']:.4f} |"
        )
    lines += ["", f"Rule: {verdict['rule']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals-dir", required=True, type=Path)
    parser.add_argument("--defense", default="krumtge",
                        help="signal-log defense token (v1.6 § 3.2: deployed primary)")
    parser.add_argument("--mode", default="flower_persistent")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    parser.add_argument("--out", type=Path, default=None,
                        help="artifact dir (results/<gate-date>/ramp_selection/)")
    args = parser.parse_args(argv)

    result = run_analysis(args.signals_dir, defense=args.defense, mode=args.mode,
                          seeds=args.seeds, out_dir=args.out)
    print(_curve_markdown(result))
    v = result["verdict"]
    print(f"VERDICT: ramp={v['selected']} switched={v['switched']} "
          f"qualifying={v['qualifying']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
