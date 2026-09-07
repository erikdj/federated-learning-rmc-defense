"""Amendment v1.8 §4 pre-unblinding lock: compute the {1%, 5%} F8 bracket cuts.

The v1.8 reporting convention (docs/reproduction/experiments.md) moves the committee-facing PRIMARY detection
operating point to recall@1%FPR and mandates a {1%, 5%, 10%} bracket everywhere
recall-at-fixed-FPR is reported. The 10% cuts are already frozen on record; this
script derives the 1% and 5% cuts from the SAME already-collected dev honest logs
by the SAME frozen F8 procedure (`h2_threshold_pipeline.select_threshold` on the
per-defense honest-only pool, all rounds). No new data, no re-selection — a pure
additional-quantile computation, computed once and recorded before unblinding.

CHAIN OF CUSTODY — the honest pools are the EXACT ones that produced the frozen
10% cuts (verified provenance, NOT the task pointer which mis-named EXP-011 for
Krum/TrustScore):

  PRIMARY (S4-only faithful scope — the adjudicated scope):
    krum, trustscore : EXP-005c + EXP-005e, S4_full_mix, seed42
                       (METHODOLOGY_LOG v-entry: "selected on EXP-005c+005e dev
                       honest scores at FPR<=0.10 (krum 0.3983 / trustscore
                       0.3575)"). Those calibration runs were S4_full_mix ONLY.
    tge, krum_tge : EXP-014 ramp-3 closed-loop, S4_full_mix, 5 dev seeds
                       (the §4 re-freeze that set the frozen 0.5754 / 0.0391).

  SENSITIVITY (S0-S4 pooled — reported, NOT adjudicated):
    tge, krum_tge : EXP-014, all S0-S4, 5 dev seeds (SAME runs, broader scope;
                       frozen sensitivity 0.2885 / 0.0562 on the closed_loop
                       record). Fully anchored.
    krum, trustscore : N/A. NO S0-S4 pool exists under the frozen-cut provenance
                       (EXP-005c/e are S4-only). Per the director ruling
                       (2026-07-26), this LOCK record carries ONLY anchor-verified
                       numbers, so the Krum/TrustScore S0-S4 sensitivity is NOT in
                       the freeze. The EXP-011 dev sweep (+EXP-012/013 refills) is
                       a DIFFERENT run/seed set with NO frozen 10% record; it can
                       be computed later as a clearly-separate non-freeze
                       diagnostic via the OPT-IN --krumts-sensitivity flag (default
                       OFF), which is flagged UNANCHORED and never adjudicated.

HARD ANCHOR GATE: every pool that HAS a frozen 10% record must reproduce it
EXACTLY (to the recorded precision). Any mismatch aborts loudly — 1%/5% numbers
are never emitted from a pool that fails its anchor.

v1.10 §3.3 PARAMETERIZATION : the input pool roots are now CLI-
parameterizable (`--source-root`, with per-pool `--dev-src` / `--exp014-signals`
overrides) so the leak-free reseal pools can be supplied without editing this
script. With everything unset the defaults are EXACTLY the frozen hard-coded
paths above (value-identity proven by the regression gate in
tests/test_compute_bracket_cuts.py). When ANY input override is active the
script refuses to write inside the frozen leak-on snapshot dir
(reproduction/protocol/h2-bracket/), which is preserved byte-for-byte.

v1.14 §3.1 LEAK-FREE RE-FREEZE MODE (`--leakfree-signals`): executes the
v1.10 §3.1-3.5 cut mechanics on the leak-free reseal pools. The
parameterization above turned out to be insufficient for this in three ways,
each fixed here and each pinned by a test:

  A. ANCHOR REGIME. FROZEN_10PCT is a leak-ON record. Asserting it against a
     leak-FREE pool aborts every run by construction. Leak-free pools are keyed
     `primary_s4_leakfree`, absent from FROZEN_10PCT, so compute_pool takes its
     existing frozen-is-None branch. The leak-on defaults keep their anchors:
     the anchor gate is regime-scoped, never weakened.
  B. WHOLE-REGIME REDIRECTION. Under `--exp014-signals` alone, krum/trustscore
     kept resolving to the leak-ON EXP-005c/e cells and PASSED their anchors —
     a SILENT cross-regime operating point, which v1.10 §2.4 forbids. Leak-free
     mode redirects ALL FOUR configs together onto the single 5-seed leak-free
     dir, and REFUSES to run alongside any leak-on input override.
  C. SENSITIVITY POOLS. v1.10 §3.2 does NOT reproduce the S0-S4 pooled
     sensitivity in the leak-free regime (it was a leak-on reported-only,
     never-adjudicated diagnostic). Leak-free mode registers the four primary
     pools and nothing else.

It also emits the two artifact-contract fields v1.10 §3.3 requires ("No cut is
emitted without these") that the leak-on path never produced: the per-quantile
degeneracy warning and the Krum+TGE survivor coverage.

The leak-free mode adds NOTHING to a default-input build — that is what keeps
the value-identity gate byte-for-byte — and it is itself an overridden-input
run, so run_identity_gate gates it exactly as it gates --source-root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_variance_envelope import DEFENSE_SCORE_SPECS, _scenario_stem  # noqa: E402
from h2_threshold_pipeline import select_threshold  # noqa: E402

# --- pre-registered bracket (v1.8 §2.2) -------------------------------------
# The frozen bracket. `--quantiles` may EXTEND this (e.g. the advisor-directed
# 2% point, 2026-07-26) without touching the frozen record: any run that adds a
# quantile still recomputes 1/5/10 and must reproduce their frozen cuts exactly.
BRACKET_FPRS = (0.01, 0.05, 0.10)

SCEN_SHORT = {
    "s0_clean_baseline": "S0", "s1_benign_churn_only": "S1",
    "s2_adaptive_switching_only": "S2", "s3_identity_reset_only": "S3",
    "s4_full_mix": "S4",
}
ALL_SCENARIOS = list(SCEN_SHORT)

DEV_SRC = REPO / "results/20260726/_dev_honest_sources"
EXP014_SIGNALS = REPO / "results/20260724/exp014_closed_loop/signals"

# --- v1.14 §3.1 leak-free re-freeze ------------------------------------------
# The regime-scoped scope key. Deliberately absent from FROZEN_10PCT so
# compute_pool takes its frozen-is-None branch (blocker A). Never add a
# leak-free entry to FROZEN_10PCT: a leak-on cut is not an anchor for a
# leak-free pool.
LEAKFREE_SCOPE = "primary_s4_leakfree"
# The v1.3 dev seeds, public and unsealed. The adjudicating cut source is the
# honest sub-population WITHIN S4 (v1.10 §3.2 "S4-only faithful scope").
LEAKFREE_SEEDS = (42, 137, 256, 314, 500)
LEAKFREE_SCENARIO = "s4_full_mix"
LEAKFREE_CONFIGS = ("krum", "trustscore", "tge", "krum_tge")
# The COMPLETE bracket the leak-free freeze must carry: the v1.8 §2.2 frozen
# {1,5,10}% plus the advisor-directed 2% point (methodology v1.27). A partial
# leak-free freeze must be impossible to emit — see resolve_leakfree_quantiles.
LEAKFREE_BRACKET = (0.01, 0.02, 0.05, 0.10)
# Regime provenance for the leak-free cells CANNOT come from the signal rows:
# the signal-log schema carries no normalize_train_only / regime field (verified
# against the real EXP-041 logs). A directory name and a filename match are not
# provenance. So the leak-free mode REQUIRES a manifest staged alongside the
# signals, built at staging time from the result JSONs (which do carry
# provenance.normalize_train_only), and validates every input cell against it.
LEAKFREE_PROVENANCE_NAME = "provenance.json"

# The frozen leak-on snapshot dir (v1.10 §3.3 item 4): preserved byte-for-byte.
# When ANY input override is active, this script REFUSES to write inside it.
FROZEN_SNAPSHOT_DIR = REPO / "reproduction/protocol/h2-bracket"
FROZEN_SNAPSHOT_JSON = FROZEN_SNAPSHOT_DIR / "bracket_cuts.json"

# The ONLY structural exclusion from the value-identity comparison:
# `_meta.frozen_json_cross_check` was added post-freeze by the --anchor-json
# extension (default None) and has no counterpart in the frozen record. Every
# other field that could legitimately differ is held to a FIELD-SPECIFIC
# expected value inside run_identity_gate (never blanket-skipped):
#   _meta.anchor_verified  -> generated value MUST be True
#   _meta.purpose          -> generated value MUST equal EXPECTED_DEFAULT_PURPOSE
#                             (the frozen record carries the pre-rewrite v1.8 §4
#                             wording; it is exempted by that expected-value
#                             assertion on the generated side, not by a skip)
#   _meta.source_override,
#   _meta.identity_gate    -> MUST be ABSENT from a default-input build
IDENTITY_SKIP_FIELDS = ("_meta.frozen_json_cross_check",)

# The current default-bracket _meta.purpose text, pinned byte-exact so any
# wording drift fails the identity gate loudly:
EXPECTED_DEFAULT_PURPOSE = (
    "F8 bracket cuts at FPR ['0.01', '0.05', '0.10'] on the frozen dev honest "
    "snapshot, per defense config (extends the frozen {1%,5%,10%} record; the "
    "10% point is the anchor)")

# Pinned structural shape of the frozen record, re-validated independently by
# main on every attestation (6 pools x {1%,5%,10%} = 18 cuts):
EXPECTED_GATE_POOLS = 6
EXPECTED_GATE_CUTS = 18


class AnchorError(RuntimeError):
    """Raised (loudly) when a pool with a frozen 10% record fails to reproduce it."""


# --- input-path resolution (v1.10 §3.3 parameterization) --------------------
def resolve_source_paths(source_root: Path | None,
                         dev_src: Path | None,
                         exp014_signals: Path | None) -> tuple[Path, Path, bool]:
    """Resolve the two input pool roots.

    Precedence: per-pool override (--dev-src / --exp014-signals) beats the
    --source-root-derived layout, which beats the frozen hard-coded defaults.
    With everything unset the returned paths are EXACTLY the frozen defaults —
    the value-identity regression gate rests on that. Every supplied override
    must resolve to an existing directory: a missing path is a wrong-pool
    custody error and fails loudly, never a silent fallback.
    """
    overridden = any(p is not None for p in (source_root, dev_src, exp014_signals))
    if source_root is not None and not source_root.is_dir():
        raise FileNotFoundError(
            f"--source-root does not exist or is not a directory: {source_root}")
    resolved_dev = dev_src if dev_src is not None else (
        source_root / "_dev_honest_sources" if source_root is not None else DEV_SRC)
    resolved_sig = exp014_signals if exp014_signals is not None else (
        source_root / "exp014_closed_loop" / "signals"
        if source_root is not None else EXP014_SIGNALS)
    if overridden:
        for label, p in (("dev honest sources (--dev-src)", resolved_dev),
                         ("EXP-014 signals (--exp014-signals)", resolved_sig)):
            if not p.is_dir():
                raise FileNotFoundError(
                    f"resolved {label} dir does not exist: {p} — supply the "
                    "per-pool override or fix the --source-root layout "
                    "(<root>/_dev_honest_sources, <root>/exp014_closed_loop/signals)")
    return resolved_dev, resolved_sig, overridden


def ensure_out_dir_allowed(out_dir: Path, out_name: str,
                           inputs_overridden: bool) -> None:
    """v1.10 §3.3 item 4: the frozen leak-on snapshot under
    reproduction/protocol/h2-bracket/ is preserved byte-for-byte. When any input
    override is active, refuse to write anywhere inside that dir — overridden-
    input outputs (e.g. the leak-free reseal) must go to an explicit, separate
    --out-dir. Default (no overrides) behavior is untouched."""
    if not inputs_overridden:
        return
    frozen = FROZEN_SNAPSHOT_DIR.resolve()
    target = (out_dir / out_name).resolve()
    if target == frozen or frozen in target.parents:
        raise AnchorError(
            f"refusing to write {target} inside the frozen leak-on snapshot dir "
            f"{frozen} while input overrides are active (v1.10 §3.3: the old "
            "leak-on artifacts are preserved byte-for-byte). Pass an --out-dir "
            "outside the frozen snapshot dir.")


# --- source registry --------------------------------------------------------
# Each pool: list of (scenario, defense, seed) -> file path, resolved below.
# The frozen 10% record each pool must reproduce EXACTLY (None => unanchored).
def _f(exp: str, scenario: str, defense: str, seed: int,
       dev_src: Path = DEV_SRC) -> Path:
    return dev_src / exp / f"{scenario}__{defense}__persistent_optimizer__seed{seed}.jsonl"


def _exp014(scenarios: list[str], defense: str,
            signals_dir: Path = EXP014_SIGNALS) -> list[Path]:
    return [signals_dir / f"{s}__{defense}__persistent_optimizer__seed{sd}.jsonl"
            for s in scenarios for sd in (42, 137, 256, 314, 500)]


def _exp011_plus_refills(defense: str, dev_src: Path = DEV_SRC) -> list[Path]:
    """The identical 50-cell S0-S4 set the official h2_dev_read used for this
    defense: EXP-011 for every cell except the two storm-killed cells, which come
    from the EXP-012/013 refills (h2_dev_read REFILL_RULE)."""
    refill = {("krum", "s3_identity_reset_only", 137): "EXP-012",
              ("trustscore", "s1_benign_churn_only", 42): "EXP-013"}
    out = []
    for s in ALL_SCENARIOS:
        for sd in (42, 137, 256, 314, 500):
            exp = refill.get((defense, s, sd), "EXP-011")
            out.append(_f(exp, s, defense, sd, dev_src))
    return out


def _leakfree(defense: str, signals_dir: Path) -> list[Path]:
    """The five leak-free S4 cells for one config, named exactly as the v1.10
    §3.3 input-manifest contract specifies. All four configs share this one
    template and one directory — that is what makes the redirection whole-regime
    (blocker B) rather than per-pool."""
    return [signals_dir / f"{LEAKFREE_SCENARIO}__{defense}__persistent_optimizer__seed{sd}.jsonl"
            for sd in LEAKFREE_SEEDS]


def build_leakfree_sources(signals_dir: Path) -> dict:
    """v1.14 §3.1 / v1.10 §3.2: the FOUR adjudicating leak-free primary pools and
    nothing else. No sensitivity pool is registered (blocker C) and no pool
    resolves outside `signals_dir` (blocker B). Every pool is unanchored by
    construction (blocker A) — see LEAKFREE_SCOPE."""
    return {
        (d, LEAKFREE_SCOPE): {
            "scope": ["S4"], "unanchored": True,
            "files": _leakfree(d, signals_dir),
            "provenance": (
                "leak-free reseal (normalize_train_only=true), S4_full_mix, 5 dev "
                f"seeds from {signals_dir} — the v1.10 §3.2 S4-only faithful scope; "
                "control_honest is a reported reference, NOT this cut source"),
        }
        for d in LEAKFREE_CONFIGS
    }


def validate_leakfree_provenance(signals_dir: Path, files: list[Path]) -> dict:
    """v1.14 §3.1 regime-provenance gate: prove every input cell IS leak-free
    BEFORE any pool is built. `is_dir` plus a filename match is not provenance
    — a leak-ON cell staged under a directory called "leakfree" would otherwise
    be pooled silently, which is the exact failure class v1.10 §2.4 forbids.

    The signal.jsonl rows carry no regime field, so the authority is the staged
    manifest (LEAKFREE_PROVENANCE_NAME), built from the EXP-041 result JSONs at
    staging time. Every check HALTs loudly and names the offending cell; absence
    of the flag is failure, never a permissive default.
    """
    manifest_path = signals_dir / LEAKFREE_PROVENANCE_NAME
    if not manifest_path.is_file():
        raise AnchorError(
            f"MISSING REGIME PROVENANCE: no provenance manifest at {manifest_path}. "
            "Leak-free cuts may not be derived from cells whose regime is merely "
            "assumed from a directory or filename (v1.10 §2.4). Stage it from the "
            "source experiment's result JSONs — see REFREEZE_PLAN_20260806.md §6.1.")
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = {c["file"]: c for c in manifest["cells"]}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise AnchorError(
            f"UNREADABLE REGIME PROVENANCE at {manifest_path}: {e}. Expected "
            '{"cells": [{"file", "source_experiment", "normalize_train_only", '
            '"image_digest"}, ...]}.') from e

    digests: set[str] = set()
    experiments: set[str] = set()
    for path in files:
        entry = entries.get(path.name)
        if entry is None:
            raise AnchorError(
                f"UNPROVENANCED CELL: {path.name} is an input to the leak-free cut "
                f"but is absent from {manifest_path}. Every cell that feeds a cut "
                "must carry recorded regime provenance.")
        flag = entry.get("normalize_train_only")
        if flag is not True:  # exact boolean True; "true"/1/None all fail
            raise AnchorError(
                f"CROSS-REGIME CELL REFUSED: {path.name} records "
                f"normalize_train_only={flag!r}, expected boolean True. A leak-ON "
                "(or unlabelled) cell cannot feed a leak-free operating point "
                "(v1.10 §2.4 never-mix-regimes).")
        digest = entry.get("image_digest")
        if not digest:
            raise AnchorError(
                f"UNPROVENANCED CELL: {path.name} records no image_digest in "
                f"{manifest_path}; the pool's regime lineage cannot be verified.")
        digests.add(digest)
        experiments.add(entry.get("source_experiment", "?"))

    if len(digests) > 1:
        raise AnchorError(
            "MIXED-IMAGE POOL REFUSED: the leak-free S4 cut source spans "
            f"{len(digests)} image digests {sorted(digests)}. The adjudicating S4 "
            "pool is single-lineage by design (v1.11 §4.2); a pool spanning an "
            "image boundary is a v1.11 §3 no-pooling violation.")

    try:
        shown = str(manifest_path.relative_to(REPO))
    except ValueError:
        shown = str(manifest_path)
    return {"manifest": shown, "cells_validated": len(files),
            "normalize_train_only": True, "regime": "leak-free",
            "image_digest": digests.pop() if digests else None,
            "source_experiments": sorted(experiments)}


def resolve_leakfree_quantiles(explicit: list[float] | None) -> list[float]:
    """v1.14 §3.1: a leak-free freeze carries the COMPLETE bracket or none at
    all. Unset => the full bracket; any other set HALTs rather than silently
    emitting a partial freeze that later reads as complete."""
    if explicit is None:
        return list(LEAKFREE_BRACKET)
    got = sorted(set(explicit))
    if got != sorted(LEAKFREE_BRACKET):
        raise AnchorError(
            f"INCOMPLETE LEAK-FREE BRACKET: --quantiles {got} != the complete "
            f"required bracket {list(LEAKFREE_BRACKET)}. The re-freeze records all "
            "four operating points together (v1.8 §2.2 {1,5,10}% + the v1.27 2% "
            "point); a partial leak-free freeze must not be emittable. Omit "
            "--quantiles to get the complete bracket.")
    return got


def survivor_coverage(files: list[Path], defense: str) -> dict:
    """v1.10 §3.3 'with survivor coverage reported'. The frozen eligibility
    filter keeps honest rows carrying a non-null score; for Krum+TGE the composed
    chain filters most rows upstream, so the pool is survivor-scoped and its
    denominator matters when reading the low quantiles. Reports the denominator
    the frozen path never emitted. Pure read; no mutation of any pool."""
    spec = DEFENSE_SCORE_SPECS[defense]
    honest = scored = 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("malicious_gt"):
                    continue
                honest += 1
                if r.get(spec.score_field) is not None:
                    scored += 1
    return {"score_field": spec.score_field,
            "honest_rows_total": honest,
            "honest_rows_scored": scored,
            "survivor_coverage": (scored / honest) if honest else None,
            "rows_dropped_null_score": honest - scored}


def degeneracy(pool_result: dict) -> dict:
    """v1.10 §3.3 degeneracy warning flag ('No cut is emitted without these').
    A cut is degenerate when the quantile lands inside a score mass point — the
    realized honest FPR at the cut is 0 while the nominal target is not, i.e. the
    cut cannot actually realize its operating point. This is Krum's known
    score-mass-at-zero behavior, and it is regime-stable: the leak-on record
    shows it too."""
    flags = {}
    for q, c in pool_result["bracket_cuts"].items():
        degen = c["realized_honest_fpr"] == 0.0 and c["fpr_target"] > 0
        flags[q] = {
            "degenerate": degen,
            "reason": ("quantile falls inside a score mass point: realized honest "
                       f"FPR is 0.0 at the nominal target {c['fpr_target']} "
                       f"(cut={c['cut']!r})") if degen else None,
        }
    return flags


def build_sources(krumts_sensitivity: bool, *,
                  dev_src: Path = DEV_SRC,
                  exp014_signals: Path = EXP014_SIGNALS) -> dict:
    src: dict[tuple[str, str], dict] = {}
    # PRIMARY — S4-only faithful scope
    for d in ("krum", "trustscore"):
        src[(d, "primary_s4")] = {
            "scope": ["S4"], "unanchored": False,
            "files": [_f("EXP-005c", "s4_full_mix", d, 42, dev_src),
                      _f("EXP-005e", "s4_full_mix", d, 42, dev_src)],
            "provenance": "EXP-005c + EXP-005e, S4_full_mix, seed42",
        }
    for d in ("tge", "krum_tge"):
        src[(d, "primary_s4")] = {
            "scope": ["S4"], "unanchored": False,
            "files": _exp014(["s4_full_mix"], d, exp014_signals),
            "provenance": "EXP-014 ramp-3 closed-loop, S4_full_mix, 5 dev seeds",
        }
    # SENSITIVITY — S0-S4 pooled
    for d in ("tge", "krum_tge"):
        src[(d, "sensitivity_s0s4")] = {
            "scope": ["S0", "S1", "S2", "S3", "S4"], "unanchored": False,
            "files": _exp014(ALL_SCENARIOS, d, exp014_signals),
            "provenance": "EXP-014 ramp-3 closed-loop, all S0-S4, 5 dev seeds (same runs, broader scope)",
        }
    if krumts_sensitivity:
        for d in ("krum", "trustscore"):
            src[(d, "sensitivity_s0s4")] = {
                "scope": ["S0", "S1", "S2", "S3", "S4"], "unanchored": True,
                "files": _exp011_plus_refills(d, dev_src),
                "provenance": ("EXP-011 dev sweep + EXP-012/013 storm refills "
                               "(50-cell h2_dev_read set); DIFFERENT run/seed set "
                               "from the S4-only frozen-cut source EXP-005c/e; "
                               "NO frozen 10% record -> UNANCHORED diagnostic"),
            }
    return src


# --- frozen 10% anchors (byte-identical to their records) -------------------
FROZEN_10PCT = {
    ("krum", "primary_s4"): 0.3983150958009648,        # thresholds_dev_honest.json
    ("trustscore", "primary_s4"): 0.3574853539466858,  # thresholds_dev_honest.json
    ("tge", "primary_s4"): 0.5754,                      # closed_loop_read.json primary_s4_only
    ("krum_tge", "primary_s4"): 0.0391,                 # closed_loop_read.json primary_s4_only
    ("tge", "sensitivity_s0s4"): 0.2885,                # closed_loop_read.json sensitivity_s0s4
    ("krum_tge", "sensitivity_s0s4"): 0.0562,           # closed_loop_read.json sensitivity_s0s4
}


def load_honest_pool(defense: str, scope_shorts: list[str], files: list[Path]) -> tuple[list[float], list[dict]]:
    """Pool honest scores by the frozen F8 convention (`_load_honest_scores`):
    every row with malicious_gt False AND a non-null score_field, ALL ROUNDS,
    per-defense. Strict row-level custody: each row's defense token and scenario
    stem must match the declared cell — a mismatch is a mislabelled log, never
    silently pooled."""
    spec = DEFENSE_SCORE_SPECS[defense]
    scope_full = {s for s, sh in SCEN_SHORT.items() if sh in scope_shorts}
    pool: list[float] = []
    inventory: list[dict] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"missing dev honest log: {path}")
        stem_scen = path.name.split("__")[0]
        if stem_scen not in scope_full:
            raise AnchorError(f"{path.name}: scenario {stem_scen} outside scope {scope_shorts}")
        exp_stem = stem_scen[0].upper() + stem_scen[1:]
        n_rows = n_honest = 0
        with open(path) as fh:
            for j, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                n_rows += 1
                tok = r.get("defense")
                if tok != spec.row_defense_token:
                    raise AnchorError(
                        f"{path.name} row {j}: defense token {tok!r} != "
                        f"{spec.row_defense_token!r} expected for {defense!r}")
                rsc = r.get("scenario")
                if rsc is None or _scenario_stem(rsc) != exp_stem:
                    raise AnchorError(
                        f"{path.name} row {j}: scenario {rsc!r} != registered {exp_stem!r}")
                val = r.get(spec.score_field)
                if not r.get("malicious_gt") and val is not None:
                    pool.append(float(val))
                    n_honest += 1
        try:
            shown = str(path.relative_to(REPO))
        except ValueError:
            shown = str(path)
        inventory.append({"file": shown, "rows": n_rows,
                          "honest_scored_in_scope": n_honest})
    return pool, inventory


def realized_fpr(pool: list[float], cut: float) -> float:
    return sum(1 for s in pool if s < cut) / len(pool) if pool else 0.0


def compute_pool(defense: str, scope_key: str, meta: dict,
                 quantiles=BRACKET_FPRS) -> dict:
    scope = meta["scope"]
    pool, inventory = load_honest_pool(defense, scope, meta["files"])
    if not pool:
        raise AnchorError(f"{defense}/{scope_key}: empty honest pool")
    n = len(pool)
    cuts = {}
    for fpr in quantiles:
        cut = select_threshold(pool, fpr)
        cuts[f"{fpr:.2f}"] = {
            "fpr_target": fpr,
            "cut": cut,
            "realized_honest_fpr": realized_fpr(pool, cut),
            # a q-quantile on n points rests on ~n*q order statistics:
            "effective_support_points": round(n * fpr, 2),
        }

    frozen = FROZEN_10PCT.get((defense, scope_key))
    anchor = {"has_frozen_record": frozen is not None, "unanchored": meta["unanchored"]}
    if frozen is not None:
        if "0.10" not in cuts:
            raise AnchorError(
                f"{defense}/{scope_key}: quantile set {sorted(cuts)} omits the 0.10 "
                f"anchor point; a pool with a frozen 10% record must recompute it.")
        computed10 = cuts["0.10"]["cut"]
        exact = abs(computed10 - frozen) <= 1e-12
        anchor.update({"frozen_10pct": frozen, "computed_10pct": computed10,
                       "match": exact})
        if not exact:
            raise AnchorError(
                f"ANCHOR FAILURE {defense}/{scope_key}: computed 10% cut "
                f"{computed10!r} != frozen {frozen!r} (pool n={n}). Refusing to "
                f"emit bracket cuts from a pool that fails its anchor.")

    return {"defense": defense, "scope": scope_key, "scope_scenarios": scope,
            "provenance": meta["provenance"], "n_honest": n,
            "bracket_cuts": cuts, "anchor": anchor,
            "source_files": inventory}


def verify_against_frozen_json(results: dict, frozen_path: Path) -> dict:
    """Cross-check every recomputed quantile that ALSO exists in the frozen
    bracket_cuts.json and assert the cut is VALUE-IDENTICAL. This proves the new
    (e.g. 2%) cut is drawn from the exact same distribution snapshot that produced
    the frozen record — not merely a same-shaped recomputation. Any mismatch
    aborts loudly; no extended-bracket record is emitted from a drifted snapshot."""
    if not frozen_path.exists():
        raise AnchorError(f"anchor json not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text())
    fcfg = frozen.get("configs", {})
    checks = []
    for d in results:
        for sc in results[d]:
            fpool = fcfg.get(d, {}).get(sc)
            if not fpool:
                continue
            fcuts = fpool["bracket_cuts"]
            ccuts = results[d][sc]["bracket_cuts"]
            for q in sorted(set(fcuts) & set(ccuts)):
                fv, cv = fcuts[q]["cut"], ccuts[q]["cut"]
                match = fv == cv
                checks.append({"config": f"{d}/{sc}", "quantile": q,
                               "frozen_cut": fv, "computed_cut": cv, "match": match})
                if not match:
                    raise AnchorError(
                        f"FROZEN-JSON ANCHOR FAILURE {d}/{sc} q={q}: recomputed cut "
                        f"{cv!r} != frozen bracket_cuts.json {fv!r}. Refusing to emit "
                        f"an extended-bracket record from a drifted snapshot.")
    try:
        shown = str(frozen_path.relative_to(REPO))
    except ValueError:
        shown = str(frozen_path)
    return {"frozen_json": shown, "shared_quantiles_checked": len(checks),
            "all_match": all(c["match"] for c in checks), "checks": checks}


def deep_diff(frozen, generated, path: str = "", *, skip: tuple = ()) -> list[str]:
    """Recursively compare two JSON-shaped objects and return the dotted paths
    of EVERY difference (value mismatch, missing key on either side, length
    mismatch). Paths listed in `skip` (exact dotted-path match) are excluded —
    the caller must enumerate them explicitly. Pure function; no mutation."""
    if isinstance(frozen, dict) and isinstance(generated, dict):
        out: list[str] = []
        for k in sorted(set(frozen) | set(generated)):
            p = f"{path}.{k}" if path else k
            if p in skip:
                continue
            if k not in frozen:
                out.append(f"{p}: missing in frozen, generated={generated[k]!r}")
            elif k not in generated:
                out.append(f"{p}: missing in generated, frozen={frozen[k]!r}")
            else:
                out.extend(deep_diff(frozen[k], generated[k], p, skip=skip))
        return out
    if isinstance(frozen, list) and isinstance(generated, list):
        if len(frozen) != len(generated):
            return [f"{path}: length {len(frozen)} (frozen) != {len(generated)} (generated)"]
        return [d for i, (a, b) in enumerate(zip(frozen, generated))
                for d in deep_diff(a, b, f"{path}[{i}]", skip=skip)]
    if frozen != generated:
        return [f"{path}: frozen={frozen!r} != generated={generated!r}"]
    return []


def run_identity_gate(frozen_path: Path = FROZEN_SNAPSHOT_JSON) -> dict:
    """v1.10 §3.3 items 2+3, enforced AT RUNTIME: recompute the DEFAULT-input
    pools and verify the COMPLETE output object is value-identical to the
    committed frozen leak-on snapshot (the only structural exclusion is
    IDENTITY_SKIP_FIELDS; all other historic deltas are held to field-specific
    expected values, asserted below). Called
    unconditionally by main whenever ANY input override is active, BEFORE any
    overridden-input (leak-free) computation — no skip path, no escape flag: a
    checkout that cannot prove default-input identity CANNOT derive leak-free
    cuts. Returns the attestation recorded in the output _meta."""
    if not frozen_path.exists():
        raise AnchorError(
            f"IDENTITY GATE FAILURE: frozen snapshot JSON not found: {frozen_path}. "
            "Cannot prove default-input value identity; refusing to compute from "
            "overridden inputs.")
    frozen = json.loads(frozen_path.read_text())
    quantiles = sorted(frozen["_meta"]["bracket_fprs"])
    for label, p in (("dev honest sources", DEV_SRC),
                     ("EXP-014 signals", EXP014_SIGNALS)):
        if not p.is_dir():
            raise AnchorError(
                f"IDENTITY GATE FAILURE: frozen default {label} dir absent: {p}. "
                "This checkout cannot reproduce the frozen snapshot from the "
                "default inputs, so it MUST NOT derive cuts from overridden "
                "inputs (v1.10 §3.3: leak-free derivation only AFTER the "
                "identity gate passes).")
    # module-level lookups (not parameter defaults) so the gate always sees the
    # live default paths:
    sources = build_sources(False, dev_src=DEV_SRC, exp014_signals=EXP014_SIGNALS)
    try:
        results = compute_results(sources, quantiles)
    except FileNotFoundError as e:
        raise AnchorError(
            f"IDENTITY GATE FAILURE: frozen default pool file absent: {e}. "
            "Refusing to compute from overridden inputs.") from e
    # normalize exactly as the file would be written, then compare EVERYTHING
    generated = json.loads(json.dumps(
        build_output(results, quantiles), allow_nan=False))
    meta_gen = generated["_meta"]

    # field-specific expected-value assertions — these
    # fields are NOT skipped; each is held to its pinned expectation:
    if meta_gen.get("anchor_verified") is not True:
        raise AnchorError(
            "IDENTITY GATE FAILURE: default-input recomputation has "
            f"_meta.anchor_verified={meta_gen.get('anchor_verified')!r}, expected "
            "True — the default build failed its own anchor certification.")
    if meta_gen.get("purpose") != EXPECTED_DEFAULT_PURPOSE:
        raise AnchorError(
            "IDENTITY GATE FAILURE: default-input _meta.purpose drifted from the "
            f"pinned text.\n  expected: {EXPECTED_DEFAULT_PURPOSE!r}\n  "
            f"generated: {meta_gen.get('purpose')!r}")
    for k in ("source_override", "identity_gate", "leakfree_provenance"):
        if k in meta_gen:
            raise AnchorError(
                f"IDENTITY GATE FAILURE: _meta.{k} present in a DEFAULT-input "
                "build — it must only ever appear under input overrides.")

    # strict deep comparison of everything else. The three asserted-above
    # fields are removed from fresh copies (never mutated in place):
    # anchor_verified only exists on the generated side; purpose exists on both
    # but with the frozen pre-rewrite wording, already pinned above.
    frozen_cmp = {**frozen, "_meta": {
        k: v for k, v in frozen["_meta"].items() if k != "purpose"}}
    generated_cmp = {**generated, "_meta": {
        k: v for k, v in meta_gen.items()
        if k not in ("purpose", "anchor_verified")}}
    mismatches = deep_diff(frozen_cmp, generated_cmp, skip=IDENTITY_SKIP_FIELDS)
    if mismatches:
        shown = "\n  ".join(mismatches[:20])
        more = "" if len(mismatches) <= 20 else f"\n  ... and {len(mismatches) - 20} more"
        raise AnchorError(
            f"IDENTITY GATE FAILURE: default-input recomputation is NOT value-"
            f"identical to {frozen_path} — {len(mismatches)} mismatching path(s):"
            f"\n  {shown}{more}\nRefusing to compute from overridden inputs.")
    try:
        shown_path = str(frozen_path.relative_to(REPO))
    except ValueError:
        shown_path = str(frozen_path)
    return {
        "passed": True,
        "compared_against": shown_path,
        "pools_checked": sum(len(v) for v in frozen["configs"].values()),
        "cuts_checked": sum(len(p["bracket_cuts"]) for d in frozen["configs"].values()
                            for p in d.values()),
        "fields_excluded": list(IDENTITY_SKIP_FIELDS),
        "field_expectations": {
            "_meta.anchor_verified": "generated == True (asserted)",
            "_meta.purpose": ("generated == pinned EXPECTED_DEFAULT_PURPOSE; "
                              "frozen carries the pre-rewrite v1.8 wording "
                              "(exempted by expected-value assertion, not skip)"),
            "_meta.source_override": "asserted ABSENT from default build",
            "_meta.identity_gate": "asserted ABSENT from default build",
            "_meta.leakfree_provenance": "asserted ABSENT from default build",
        },
    }


def compute_results(sources: dict, quantiles) -> dict:
    """Compute every registered pool in the frozen presentation order."""
    order = [("krum", "primary_s4"), ("trustscore", "primary_s4"),
             ("tge", "primary_s4"), ("krum_tge", "primary_s4"),
             ("tge", "sensitivity_s0s4"), ("krum_tge", "sensitivity_s0s4"),
             ("krum", "sensitivity_s0s4"), ("trustscore", "sensitivity_s0s4"),
             # leak-free pools (v1.14 §3.1). Never present in a default-input
             # run, so the frozen presentation order above is untouched.
             *((d, LEAKFREE_SCOPE) for d in LEAKFREE_CONFIGS)]
    results: dict = {}
    for key in order:
        if key not in sources:
            continue
        d, sc = key
        results.setdefault(d, {})[sc] = compute_pool(d, sc, sources[key], quantiles)
    return results


def build_output(results: dict, quantiles, frozen_json_check: dict | None = None,
                 source_override: dict | None = None,
                 identity_gate: dict | None = None,
                 purpose_override: str | None = None,
                 unanchored_record: bool = False,
                 leakfree_provenance: dict | None = None) -> dict:
    """Assemble the output object. For a default run (no cross-check, no
    override, no gate, no purpose override) this is byte-identical to the
    pre-parameterization script's output — the identity gate and the regression
    tests rest on that. `purpose_override` exists so the leak-free record does
    not inherit the leak-on wording ("on the frozen dev honest snapshot"), which
    would be a false provenance statement in a leak-free artifact."""
    q_keys = [f"{q:.2f}" for q in quantiles]
    anchors_checked = [(d, sc) for d in results for sc in results[d]
                       if results[d][sc]["anchor"]["has_frozen_record"]]
    all_ok = all(results[d][sc]["anchor"]["match"] for d, sc in anchors_checked)
    anchor_verified = all_ok and (frozen_json_check is None
                                  or frozen_json_check["all_match"])
    # An unanchored record must NOT claim anchor_verified:true just because zero
    # anchors were checked — that would read as a verification that never ran.
    # `None` + an explicit reason; the identity gate is recorded separately as
    # the verification that DID run.
    anchor_gate_text = ("every pool with a frozen 10% record reproduces it EXACTLY"
                        if not unanchored_record else
                        "UNANCHORED BY CONSTRUCTION — no anchor assertion was made or "
                        "could be: the frozen 10% record is a leak-on value and is not "
                        "an anchor for the leak-free regime (v1.14 §3.1, v1.10 §2.4 "
                        "never-mix-regimes). See identity_gate for the verification "
                        "that did run.")
    if unanchored_record:
        anchor_verified = None
    return {
        "_meta": {
            "purpose": purpose_override if purpose_override is not None else
                       f"F8 bracket cuts at FPR {q_keys} on the frozen dev honest "
                       "snapshot, per defense config (extends the frozen "
                       "{1%,5%,10%} record; the 10% point is the anchor)",
            "spec": "docs/superpowers/specs/2026-07-26-praxis-experimental-design-v1.8.md §2-§4",
            "procedure": "h2_threshold_pipeline.select_threshold on per-defense "
                         "honest-only pool (malicious_gt False, non-null score, ALL "
                         "ROUNDS); same frozen F8 procedure, computed at each bracket FPR",
            "bracket_fprs": quantiles,
            "primary_scope": "S4-only faithful (adjudicated)",
            "sensitivity_scope": "S0-S4 pooled (reported, not adjudicated)",
            "anchor_gate": anchor_gate_text,
            "anchor_gate_passed": all_ok,
            "anchors_checked": [f"{d}/{sc}" for d, sc in anchors_checked],
            "frozen_json_cross_check": frozen_json_check,
            "anchor_verified": anchor_verified,
            # only present when inputs were overridden — the default (unset)
            # output stays byte-identical to the pre-parameterization script:
            **({"source_override": source_override}
               if source_override is not None else {}),
            **({"identity_gate": identity_gate}
               if identity_gate is not None else {}),
            **({"leakfree_provenance": leakfree_provenance}
               if leakfree_provenance is not None else {}),
        },
        "configs": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "reproduction/protocol/h2-bracket")
    ap.add_argument("--out-name", default="bracket_cuts.json",
                    help="output filename within --out-dir")
    ap.add_argument("--quantiles", type=float, nargs="+", default=None,
                    help="bracket FPRs to compute; MUST include 0.10 (the anchor) "
                         "and reproduce the frozen 1/5/10 cuts. Unset = the frozen "
                         "bracket (0.01 0.05 0.10); pass e.g. `0.01 0.02 0.05 0.10` "
                         "to add the 2%% point. In --leakfree-signals mode the "
                         "COMPLETE bracket 0.01 0.02 0.05 0.10 is required (it is "
                         "the default there); any other set HALTs.")
    ap.add_argument("--anchor-json", type=Path, default=None,
                    help="frozen bracket_cuts.json to cross-verify shared quantiles "
                         "against (value-identical); proves same distribution snapshot")
    # v1.10 §3.3 parameterization : supply the leak-free pool paths
    # without editing this script. Unset => the frozen hard-coded defaults,
    # value-identically (regression-gated in tests/test_compute_bracket_cuts.py).
    ap.add_argument("--source-root", type=Path, default=None,
                    help="root dir for ALL input pools, laid out as "
                         "<root>/_dev_honest_sources/<EXP-NNN>/... and "
                         "<root>/exp014_closed_loop/signals/...; unset = the "
                         "frozen default paths (must exist; fails loudly if not)")
    ap.add_argument("--dev-src", type=Path, default=None,
                    help="per-pool override for the dev honest sources dir "
                         "(EXP-005c/e + refills); beats --source-root")
    ap.add_argument("--exp014-signals", type=Path, default=None,
                    help="per-pool override for the EXP-014 closed-loop signals "
                         "dir; beats --source-root")
    # v1.14 §3.1 leak-free re-freeze mode. Mutually exclusive with every
    # leak-on input override (see the cross-regime guard in main).
    ap.add_argument("--leakfree-signals", type=Path, default=None,
                    help="LEAK-FREE RE-FREEZE (v1.14 §3.1): dir holding the "
                         "leak-free reseal S4 cells "
                         "s4_full_mix__<config>__persistent_optimizer__seed<N>.jsonl "
                         "for all four configs x 5 dev seeds. Registers ONLY the "
                         "four unanchored primary_s4_leakfree pools (no S0-S4 "
                         "sensitivity, per v1.10 §3.2) and emits the degeneracy + "
                         "survivor-coverage artifact-contract fields. Cannot be "
                         "combined with any leak-on input override.")
    # Default OFF (director ruling 2026-07-26): this is a pre-unblinding LOCK
    # record and every number in it must be anchor-verified. The Krum/TrustScore
    # S0-S4 sensitivity has no same-provenance pool (EXP-005c/e are S4-only), so
    # it is NOT part of the freeze. The EXP-011-derived number remains available
    # as an OPT-IN, clearly-separate non-freeze diagnostic (--krumts-sensitivity).
    ap.add_argument("--krumts-sensitivity", action="store_true", default=False,
                    help="OPT-IN: also compute the UNANCHORED EXP-011-derived "
                         "Krum/TrustScore S0-S4 sensitivity (a separate non-freeze "
                         "diagnostic; NOT part of the locked bracket record)")
    args = ap.parse_args()

    # v1.14 §3.1 R2: leak-free requires the COMPLETE bracket; leak-on keeps its
    # frozen default byte-for-byte when --quantiles is unset.
    if args.leakfree_signals is not None:
        quantiles = resolve_leakfree_quantiles(args.quantiles)
    else:
        quantiles = sorted(set(args.quantiles if args.quantiles is not None
                               else BRACKET_FPRS))
    q_keys = [f"{q:.2f}" for q in quantiles]

    # v1.14 §3.1 leak-free mode. The cross-regime guard is FIRST: mixing the
    # leak-free source with any leak-on input override is exactly the silent
    # cross-regime operating point v1.10 §2.4 forbids (blocker B), and
    # reproducing a pooled S0-S4 sensitivity leak-free is refused by v1.10 §3.2
    # (blocker C). Both HALT rather than quietly producing a defensible-looking
    # artifact.
    leakfree = args.leakfree_signals
    if leakfree is not None:
        conflicting = [n for n, v in (("--source-root", args.source_root),
                                      ("--dev-src", args.dev_src),
                                      ("--exp014-signals", args.exp014_signals))
                       if v is not None]
        if conflicting:
            raise AnchorError(
                f"--leakfree-signals cannot be combined with {', '.join(conflicting)}: "
                "that would derive cuts from a cross-regime pool (leak-free S4 cells "
                "alongside leak-on sources), which v1.10 §2.4 forbids. The leak-free "
                "re-freeze redirects ALL FOUR configs onto the leak-free dir together.")
        if args.krumts_sensitivity:
            raise AnchorError(
                "--krumts-sensitivity is not available in leak-free mode: v1.10 §3.2 "
                "does NOT reproduce the S0-S4 pooled sensitivity in the leak-free "
                "regime (it was a leak-on reported-only, never-adjudicated "
                "diagnostic). A leak-free pooled sensitivity would require explicitly "
                "enumerated leak-free S0-S4 honest cells, which do not exist.")
        if not leakfree.is_dir():
            raise FileNotFoundError(
                f"--leakfree-signals does not exist or is not a directory: {leakfree}")
        dev_src, exp014_signals, inputs_overridden = DEV_SRC, EXP014_SIGNALS, True
    else:
        dev_src, exp014_signals, inputs_overridden = resolve_source_paths(
            args.source_root, args.dev_src, args.exp014_signals)
    ensure_out_dir_allowed(args.out_dir, args.out_name, inputs_overridden)

    # v1.10 §3.3 item 3, enforced structurally: ANY input override (the leak-
    # free path) FIRST proves the DEFAULT inputs still reproduce the frozen
    # snapshot value-identically. Hard-aborts (AnchorError) if the default
    # pools are absent or any value mismatches — no skip path, no escape flag.
    # ACCEPTED LIMITATION: this gate
    # defends against OPERATOR ERROR (wrong checkout, absent pools, accidental
    # derivation without verification). A deliberate in-process actor with
    # import access could rebind this call (or deep_diff, open, the frozen
    # JSON itself) — no in-module CPython construction resists that actor;
    # such an adversary is out of scope for this and every other file-based
    # integrity control in the repo.
    identity_gate = None
    if inputs_overridden:
        identity_gate = run_identity_gate()
        # Independent re-validation in main's own control flow: main does not blindly trust the returned attestation —
        # it must be structurally complete, passed, and reference the
        # hardcoded frozen snapshot with the pinned pool/cut counts.
        expected_vs = str(FROZEN_SNAPSHOT_JSON.relative_to(REPO))
        if (not isinstance(identity_gate, dict)
                or identity_gate.get("passed") is not True
                or identity_gate.get("compared_against") != expected_vs
                or identity_gate.get("pools_checked") != EXPECTED_GATE_POOLS
                or identity_gate.get("cuts_checked") != EXPECTED_GATE_CUTS):
            raise AnchorError(
                "IDENTITY GATE ATTESTATION INVALID: expected passed=True, "
                f"compared_against={expected_vs!r}, pools_checked="
                f"{EXPECTED_GATE_POOLS}, cuts_checked={EXPECTED_GATE_CUTS}; "
                f"got {identity_gate!r}. Refusing to proceed.")
        print(f"== IDENTITY GATE (default inputs vs "
              f"{identity_gate['compared_against']}) == PASSED "
              f"({identity_gate['pools_checked']} pools / "
              f"{identity_gate['cuts_checked']} cuts value-identical)")

    leakfree_provenance = None
    if leakfree is not None:
        sources = build_leakfree_sources(leakfree)
        # R1: prove every input cell IS leak-free BEFORE any pool is read.
        leakfree_provenance = validate_leakfree_provenance(
            leakfree, [f for m in sources.values() for f in m["files"]])
        print(f"== REGIME PROVENANCE ({leakfree_provenance['manifest']}) == "
              f"PASSED ({leakfree_provenance['cells_validated']} cells, "
              f"normalize_train_only=True, single image lineage "
              f"{leakfree_provenance['image_digest']}, "
              f"source {', '.join(leakfree_provenance['source_experiments'])})")
        purpose_override = (
            f"LEAK-FREE re-frozen F8 bracket cuts at FPR {q_keys}, per defense "
            "config, on the leak-free reseal honest pool (normalize_train_only="
            "true) within S4_full_mix (v1.10 §3.2 S4-only faithful scope). "
            "Same frozen procedure, leak-free inputs; UNANCHORED by construction "
            "— the frozen 10% record is a leak-on value and is not an anchor for "
            "this regime (v1.14 §3.1, v1.10 §2.4 never-mix-regimes).")
        override_record = {"leakfree_signals": str(leakfree),
                           "regime": "leak-free (normalize_train_only=true)"}
    else:
        sources = build_sources(args.krumts_sensitivity,
                                dev_src=dev_src, exp014_signals=exp014_signals)
        purpose_override = None
        override_record = ({"dev_src": str(dev_src),
                            "exp014_signals": str(exp014_signals)}
                           if inputs_overridden else None)
    results = compute_results(sources, quantiles)

    # v1.10 §3.3 artifact contract: "No cut is emitted without these."
    if leakfree is not None:
        for d in results:
            for sc in results[d]:
                results[d][sc]["degeneracy"] = degeneracy(results[d][sc])
                results[d][sc]["survivor_coverage"] = survivor_coverage(
                    sources[(d, sc)]["files"], d)

    frozen_json_check = None
    if args.anchor_json is not None:
        frozen_json_check = verify_against_frozen_json(results, args.anchor_json)

    out = build_output(
        results, quantiles, frozen_json_check=frozen_json_check,
        source_override=override_record,
        identity_gate=identity_gate, purpose_override=purpose_override,
        unanchored_record=leakfree is not None,
        leakfree_provenance=leakfree_provenance)
    anchors_checked = [tuple(s.split("/")) for s in out["_meta"]["anchors_checked"]]
    all_ok = out["_meta"]["anchor_gate_passed"]
    anchor_verified = out["_meta"]["anchor_verified"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))

    # --- console summary ----------------------------------------------------
    if leakfree is not None:
        print("== ANCHOR GATE == NOT APPLICABLE (unanchored by construction: the "
              "frozen 10% record is leak-on and is not an anchor for this regime; "
              "the identity gate above is the verification that ran)")
    else:
        print("== ANCHOR GATE (10% must reproduce frozen record EXACTLY) ==")
        for d, sc in anchors_checked:
            a = results[d][sc]["anchor"]
            print(f"  {d}/{sc}: computed {a['computed_10pct']!r} vs frozen "
                  f"{a['frozen_10pct']!r}  {'PASS' if a['match'] else 'FAIL'}")
        print(f"  => anchor gate {'PASSED' if all_ok else 'FAILED'}")
    if frozen_json_check is not None:
        print(f"== FROZEN-JSON CROSS-CHECK ({frozen_json_check['shared_quantiles_checked']} "
              f"shared quantiles vs {frozen_json_check['frozen_json']}) ==")
        print(f"  => {'ALL MATCH' if frozen_json_check['all_match'] else 'MISMATCH'}")
    print(f"== BRACKET CUTS ({' / '.join(q_keys)} FPR) ==")
    for d in results:
        for sc in results[d]:
            r = results[d][sc]
            c = r["bracket_cuts"]
            flag = " [UNANCHORED]" if r["anchor"]["unanchored"] else ""
            cutstr = "  ".join(f"{k}={c[k]['cut']!r}" for k in q_keys if k in c)
            print(f"  {d}/{sc} (n={r['n_honest']}){flag}: {cutstr}")
            degen = sorted(q for q, f in r.get("degeneracy", {}).items()
                           if f["degenerate"])
            if degen:
                print(f"      [DEGENERATE at {', '.join(degen)} FPR — quantile "
                      "inside a score mass point; realized honest FPR is 0.0]")
            cov = r.get("survivor_coverage")
            if cov is not None and cov["rows_dropped_null_score"]:
                print(f"      [survivor-scoped: {cov['honest_rows_scored']}/"
                      f"{cov['honest_rows_total']} honest rows carry "
                      f"{cov['score_field']} (coverage {cov['survivor_coverage']:.4f})]")
    print(f"wrote {out_path}")
    # anchor_verified is None ONLY on the unanchored leak-free record, where the
    # gates that CAN fail (identity, regime provenance, bracket completeness)
    # have already hard-raised. It is not a failure signal.
    return 0 if (anchor_verified is None or anchor_verified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
