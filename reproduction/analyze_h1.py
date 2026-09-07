"""H1 signal-family held-out read at a development-frozen operating point.

Protocol
--------
1. Load the frozen family detectors {S, W, C} (leak-free 25-cell refit).
2. Score the 25 development cells (5 scenarios x 5 dev seeds) of the Krum+TGE
   leak-free corpus, restrict each cell to its cold-start window (first k rounds
   per identity), pool the honest cold-start scores per family ACROSS the 25 dev
   cells, and set one cut per family at the 1 - target_fpr quantile of that pool
   (``select_threshold_risk``; flag test is ``score > cut``).  These cuts are
   frozen before the held-out corpus is touched.
3. Read the held-out Krum+TGE corpus per (scenario, seed) cell.  Cells are scored
   one at a time because ``logical_cid`` values collide across seeds; pooling
   seeds would corrupt the per-identity cold-start window.
   - PRIMARY read: apply the development-frozen cut unchanged.
   - SECONDARY read ("matched operating point"): re-calibrate the cut inside each
     held-out cell from that cell's own honest cold-start scores.
4. Aggregate per scenario (mean over the held-out seeds), report the C-S and C-W
   margins, and adjudicate the base-spec § 6.1 bands for both reads:
     (a) C recall exceeds S by >= 5pp in S2/S3/S4;
     (b) C recall exceeds W by >= 5pp in S2/S3/S4;
     (c) one-sided exact paired Wilcoxon (C > family) p <= CRITERION_ALPHA
         over the held-out seeds, for both contrasts, in S2/S3/S4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO / "scripts"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_h2_dev_read import CRITERION_ALPHA, exact_wilcoxon_onesided  # noqa: E402
from h1_signal_family_eval import (  # noqa: E402
    _compute_recall_at_fpr,
    score_and_scope,
    select_threshold_risk,
)

SCENARIOS = [
    "s0_clean_baseline",
    "s1_benign_churn_only",
    "s2_adaptive_switching_only",
    "s3_identity_reset_only",
    "s4_full_mix",
]
SHORT = {s: f"S{i}" for i, s in enumerate(SCENARIOS)}
FAMILIES = ("S", "W", "C")
DEFENSE = "krum_tge"
EXEC_MODE = "persistent_optimizer"
DOMINANCE = ["S2", "S3", "S4"]
MARGIN = 0.05
EXPECTED_MODEL_SHA256 = {
    "S": "9638fede835cc72498e503271f5090ae00c47ae6fe2cd24d1dd01aecebc305f5",
    "W": "590e57ee788431fff704fa3724fa5f16bb984f0df364f42573fab499e75c26ae",
    "C": "9271838d7d14c513f731def94c16725a242bf012cce4bb28f3c18d02639e71a2",
}


# ---------------------------------------------------------------------------
# corpus enumeration
# ---------------------------------------------------------------------------

def enumerate_cells(root: Path) -> Dict[Tuple[str, int], Path]:
    """Map (scenario, seed) -> signal-log path for the deployed-config files."""
    cells: Dict[Tuple[str, int], Path] = {}
    for path in sorted(root.glob("*.jsonl")):
        scenario, defense, mode, seed_tok = path.stem.rsplit("__", 3)
        if defense != DEFENSE:
            continue
        if mode != EXEC_MODE or not seed_tok.startswith("seed"):
            raise ValueError(f"unexpected unit stem: {path.stem}")
        if scenario not in SHORT:
            raise ValueError(f"unknown scenario in {path.name}")
        cells[(scenario, int(seed_tok[4:]))] = path
    if not cells:
        raise SystemExit(f"no {DEFENSE} signal logs under {root}")
    return cells


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_classifiers(models_dir: Path, k: int) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Load the published H1 models after verifying their custody hashes."""
    if k != 3:
        raise SystemExit("the published H1 model custody supports only k=3")
    classifiers: Dict[str, Any] = {}
    model_sha: Dict[str, str] = {}
    for family in FAMILIES:
        path = models_dir / f"{family}_k{k}_final.pkl"
        actual = sha256(path)
        expected = EXPECTED_MODEL_SHA256[family]
        if actual != expected:
            raise SystemExit(
                f"{family} model hash mismatch: expected {expected}, "
                f"found {actual} at {path}"
            )
        with open(path, "rb") as stream:
            classifiers[family] = pickle.load(stream)["model"]
        model_sha[family] = actual
    return classifiers, model_sha


# ---------------------------------------------------------------------------
# stage 1 — freeze one cut per family on the development corpus
# ---------------------------------------------------------------------------

def freeze_dev_thresholds(
    clfs: Dict[str, Any],
    dev_cells: Dict[Tuple[str, int], Path],
    k: int,
    target_fpr: float,
) -> Dict[str, Dict[str, Any]]:
    pooled: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    for key in sorted(dev_cells):
        rows = load_rows(dev_cells[key])
        for fam in FAMILIES:
            _all, scoped = score_and_scope(clfs[fam], rows, fam, k)
            pooled[fam].extend(r["score"] for r in scoped if not r["malicious"])

    frozen: Dict[str, Dict[str, Any]] = {}
    for fam in FAMILIES:
        scores = pooled[fam]
        cut = select_threshold_risk(scores, target_fpr)
        realized = (
            sum(1 for s in scores if s > cut) / len(scores) if scores else None
        )
        frozen[fam] = {
            "threshold": cut,
            "n_honest_dev": len(scores),
            "target_fpr": target_fpr,
            "realized_dev_fpr": realized,
        }
    return frozen


# ---------------------------------------------------------------------------
# stage 2 — held-out read + aggregation
# ---------------------------------------------------------------------------

def aggregate(
    percell: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]],
    seeds: List[int],
) -> Dict[str, Any]:
    """Per-scenario means, margins, Wilcoxon, and the § 6.1 verdict."""

    def agg(scenario: str, fam: str, key: str) -> Optional[float]:
        vals = [
            percell[(scenario, sd)][fam][key]
            for sd in seeds
            if percell[(scenario, sd)][fam][key] is not None
        ]
        return mean(vals) if vals else None

    per_scenario: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        rec = {fam: agg(scenario, fam, "recall_at_fpr") for fam in FAMILIES}
        fpr = {fam: agg(scenario, fam, "actual_fpr") for fam in FAMILIES}
        per_scenario[SHORT[scenario]] = {
            "S": rec["S"],
            "W": rec["W"],
            "C": rec["C"],
            "C_minus_S": rec["C"] - rec["S"],
            "C_minus_W": rec["C"] - rec["W"],
            "S_actual_fpr": fpr["S"],
            "W_actual_fpr": fpr["W"],
            "C_actual_fpr": fpr["C"],
        }

    wilcoxon: Dict[str, Any] = {}
    for short in DOMINANCE:
        scenario = next(s for s, sh in SHORT.items() if sh == short)
        d_cs: List[float] = []
        d_cw: List[float] = []
        for sd in seeds:
            cell = percell[(scenario, sd)]
            c = cell["C"]["recall_at_fpr"]
            s = cell["S"]["recall_at_fpr"]
            w = cell["W"]["recall_at_fpr"]
            # Recalls are k/n_mal fractions, so equal detection-count
            # differences must be EXACT ties for the sign-rank test; round
            # away last-ulp float noise before ranking (n_mal <= a few
            # hundred, so distinct differences stay >= 1e-6 apart).
            if None not in (c, s):
                d_cs.append(round(c - s, 9))
            if None not in (c, w):
                d_cw.append(round(c - w, 9))
        w_cs, p_cs = exact_wilcoxon_onesided(d_cs)
        w_cw, p_cw = exact_wilcoxon_onesided(d_cw)
        wilcoxon[short] = {
            "p_C_gt_S": p_cs,
            "p_C_gt_W": p_cw,
            "W_plus_C_gt_S": w_cs,
            "W_plus_C_gt_W": w_cw,
            "n_pairs": len(d_cs),
            "pass": p_cs <= CRITERION_ALPHA and p_cw <= CRITERION_ALPHA,
        }

    a_pass = all(per_scenario[s]["C_minus_S"] >= MARGIN for s in DOMINANCE)
    b_pass = all(per_scenario[s]["C_minus_W"] >= MARGIN for s in DOMINANCE)
    c_pass = all(wilcoxon[s]["pass"] for s in DOMINANCE)
    verdict = {
        "a_C_gt_S_5pp_all_S2S3S4": a_pass,
        "b_C_gt_W_5pp_all_S2S3S4": b_pass,
        "c_wilcoxon_all": c_pass,
        "criterion_alpha": CRITERION_ALPHA,
        "H1": (
            "H1 CONFIRMED (held-out)"
            if (a_pass and b_pass and c_pass)
            else "H1 NOT MET (held-out) — falsification condition fired"
        ),
    }
    return {"per_scenario": per_scenario, "wilcoxon": wilcoxon, "verdict": verdict}


def print_table(title: str, block: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    hdr = (f"{'scen':<4} {'S_rec':>6} {'W_rec':>6} {'C_rec':>6} {'C-S':>7} "
           f"{'C-W':>7} {'S_fpr':>6} {'W_fpr':>6} {'C_fpr':>6}")
    print(hdr)
    for short in [SHORT[s] for s in SCENARIOS]:
        m = block["per_scenario"][short]
        print(f"{short:<4} {m['S']:>6.3f} {m['W']:>6.3f} {m['C']:>6.3f} "
              f"{m['C_minus_S']:>+7.3f} {m['C_minus_W']:>+7.3f} "
              f"{m['S_actual_fpr']:>6.3f} {m['W_actual_fpr']:>6.3f} "
              f"{m['C_actual_fpr']:>6.3f}")
    for short in DOMINANCE:
        w = block["wilcoxon"][short]
        print(f"  {short} Wilcoxon: p(C>S)={w['p_C_gt_S']:.5f} "
              f"p(C>W)={w['p_C_gt_W']:.5f} n={w['n_pairs']}")
    print("  VERDICT:", json.dumps(block["verdict"]))


# ---------------------------------------------------------------------------
# READ.md
# ---------------------------------------------------------------------------

def fmt(v: Optional[float], places: int = 3, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{places}f}" if signed else f"{v:.{places}f}"


def write_readme(path: Path, payload: Dict[str, Any]) -> None:
    meta = payload["_meta"]
    lines: List[str] = []
    lines.append("# H1 signal-family held-out read")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "The frozen family detectors S (single-round features), W (short-window "
        "temporal features) and C (their combination) are the leak-free 25-cell "
        f"refits under `{meta['models_dir']}`. Each detector emits a "
        "malicious-risk probability, so the suspicious end of the score is HIGH "
        "and a client-round is flagged when its score exceeds the operating cut. "
        f"The cut for each family is set at the {1 - meta['target_fpr']:.2f} "
        "quantile of honest scores, which targets a "
        f"{meta['target_fpr']:.0%} false-positive rate."
    )
    lines.append("")
    lines.append(
        "The operating cut is fixed on development data before the held-out "
        f"corpus is read. The {meta['n_dev_cells']} development cells "
        f"({meta['k']}-round cold-start window, deployed Krum+TGE configuration) "
        "are scored one cell at a time, their honest cold-start scores are pooled "
        "across cells, and one cut per family is taken from that pool. The "
        f"held-out corpus is then read as {meta['n_heldout_cells']} "
        "(scenario, seed) cells. Cells are scored separately because client "
        "identifiers repeat across seeds and pooling them would corrupt the "
        "per-identity cold-start window."
    )
    lines.append("")
    lines.append(
        "Two operating points are reported. The primary read applies the "
        "development-frozen cut unchanged to every held-out cell. The secondary "
        "read, labelled the matched operating point, re-derives the cut inside "
        "each held-out cell from that cell's own honest cold-start scores, so "
        "each cell is held at its own nominal false-positive rate. Recall and "
        "realized false-positive rate are computed per cell and averaged over "
        "the held-out seeds within a scenario."
    )
    lines.append("")
    lines.append(
        "Adjudication follows the base-spec § 6.1 bands: C must exceed S by at "
        f"least {MARGIN:.2f} and W by at least {MARGIN:.2f} in S2, S3 and S4, "
        "and a one-sided exact paired Wilcoxon signed-rank test over the "
        "held-out seeds must reach "
        f"p <= {CRITERION_ALPHA:.5f} for both contrasts in those scenarios."
    )
    lines.append("")

    lines.append("## Development operating cuts")
    lines.append("")
    lines.append("| Family | Cut | Honest dev rows | Target FPR | Realized dev FPR |")
    lines.append("| --- | --- | --- | --- | --- |")
    for fam in FAMILIES:
        t = payload["dev_thresholds"][fam]
        lines.append(
            f"| {fam} | {t['threshold']:.6f} | {t['n_honest_dev']} | "
            f"{t['target_fpr']:.2f} | {fmt(t['realized_dev_fpr'])} |"
        )
    lines.append("")

    for label, key in (("Primary read (development-frozen cut)", "primary"),
                       ("Secondary read (matched operating point)", "secondary")):
        block = payload[key]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| Scenario | S recall | W recall | C recall | C-S | C-W | "
            "S FPR | W FPR | C FPR |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for scenario in SCENARIOS:
            m = block["per_scenario"][SHORT[scenario]]
            lines.append(
                f"| {SHORT[scenario]} | {fmt(m['S'])} | {fmt(m['W'])} | "
                f"{fmt(m['C'])} | {fmt(m['C_minus_S'], signed=True)} | "
                f"{fmt(m['C_minus_W'], signed=True)} | {fmt(m['S_actual_fpr'])} | "
                f"{fmt(m['W_actual_fpr'])} | {fmt(m['C_actual_fpr'])} |"
            )
        lines.append("")
        lines.append("| Scenario | p (C > S) | p (C > W) | pairs | band met |")
        lines.append("| --- | --- | --- | --- | --- |")
        for short in DOMINANCE:
            w = block["wilcoxon"][short]
            lines.append(
                f"| {short} | {w['p_C_gt_S']:.5f} | {w['p_C_gt_W']:.5f} | "
                f"{w['n_pairs']} | {'yes' if w['pass'] else 'no'} |"
            )
        lines.append("")
        v = block["verdict"]
        lines.append(
            f"Band (a) C over S by {MARGIN:.2f} in S2/S3/S4: "
            f"{'met' if v['a_C_gt_S_5pp_all_S2S3S4'] else 'not met'}. "
            f"Band (b) C over W by {MARGIN:.2f} in S2/S3/S4: "
            f"{'met' if v['b_C_gt_W_5pp_all_S2S3S4'] else 'not met'}. "
            f"Band (c) Wilcoxon in S2/S3/S4: "
            f"{'met' if v['c_wilcoxon_all'] else 'not met'}."
        )
        lines.append("")
        lines.append(f"Verdict: {v['H1']}")
        lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev-signals", required=True, type=Path,
                    help="directory of development-seed Krum+TGE signal logs")
    ap.add_argument("--heldout-signals", required=True, type=Path,
                    help="directory of held-out-seed Krum+TGE signal logs")
    ap.add_argument("--models", required=True, type=Path,
                    help="directory holding {S,W,C}_k3_final.pkl")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for the read artifacts")
    ap.add_argument("--k", type=int, default=3, help="cold-start window size")
    ap.add_argument("--target-fpr", type=float, default=0.10)
    args = ap.parse_args()

    clfs, model_sha = load_classifiers(args.models, args.k)

    dev_cells = enumerate_cells(args.dev_signals)
    heldout_cells = enumerate_cells(args.heldout_signals)
    # Cohort gates: the development cohort is the five registered development
    # seeds (data/seeds.json) and the held-out cohort is the registered
    # H2-confirmatory seed set (data/h2_confirm_seeds.json); both must be
    # complete scenario x seed grids and disjoint. Derived from the registered
    # files, not from whatever happens to be on disk.
    with open(REPO / "data" / "seeds.json", encoding="utf-8") as fh:
        registered_dev = sorted(json.load(fh)["dev_seeds"])
    with open(REPO / "data" / "h2_confirm_seeds.json", encoding="utf-8") as fh:
        registered_heldout = sorted(json.load(fh)["h2_confirm_seeds"])
    dev_seeds = sorted({sd for _sc, sd in dev_cells})
    heldout_seeds = sorted({sd for _sc, sd in heldout_cells})
    if dev_seeds != registered_dev:
        raise SystemExit("development cohort does not match data/seeds.json dev_seeds")
    if heldout_seeds != registered_heldout:
        raise SystemExit("held-out cohort does not match data/h2_confirm_seeds.json")
    if set(dev_seeds) & set(heldout_seeds):
        raise SystemExit("development and held-out cohorts overlap")
    for name, cells, seeds in (("development", dev_cells, dev_seeds),
                               ("held-out", heldout_cells, heldout_seeds)):
        missing = [(sc, sd) for sc in SCENARIOS for sd in seeds
                   if (sc, sd) not in cells]
        extra = [k for k in cells if k[0] not in SCENARIOS or k[1] not in seeds]
        if missing or extra:
            raise SystemExit(f"{name} matrix incomplete or over-full: "
                             f"{len(missing)} missing, {len(extra)} extra")
    print(f"[h1_corrected_read] dev cells: {len(dev_cells)}; "
          f"held-out cells: {len(heldout_cells)} "
          f"({len(SCENARIOS)} scenarios x {len(heldout_seeds)} seeds)")

    frozen = freeze_dev_thresholds(clfs, dev_cells, args.k, args.target_fpr)
    for fam in FAMILIES:
        t = frozen[fam]
        print(f"  dev cut {fam}: {t['threshold']:.6f} "
              f"(honest n={t['n_honest_dev']}, realized dev FPR="
              f"{fmt(t['realized_dev_fpr'])})")

    primary_cells: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    secondary_cells: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        for sd in heldout_seeds:
            rows = load_rows(heldout_cells[(scenario, sd)])
            primary_cells[(scenario, sd)] = {
                fam: _compute_recall_at_fpr(
                    clfs[fam], rows, fam, args.k, args.target_fpr,
                    threshold=frozen[fam]["threshold"])
                for fam in FAMILIES
            }
            secondary_cells[(scenario, sd)] = {
                fam: _compute_recall_at_fpr(
                    clfs[fam], rows, fam, args.k, args.target_fpr)
                for fam in FAMILIES
            }

    primary = aggregate(primary_cells, heldout_seeds)
    secondary = aggregate(secondary_cells, heldout_seeds)
    print_table("PRIMARY (development-frozen cut)", primary)
    print_table("SECONDARY (matched operating point)", secondary)

    def cellmap(percell):
        return {
            f"{SHORT[sc]}__seed{sd}": {fam: percell[(sc, sd)][fam] for fam in FAMILIES}
            for sc in SCENARIOS for sd in heldout_seeds
        }

    payload = {
        "_meta": {
            "read": "H1 signal-family held-out read; frozen leak-free 25-cell "
                    "detectors; development-frozen operating cut (primary) and "
                    "per-cell matched operating point (secondary); "
                    "per-(scenario, seed) cell scoring (collision-free)",
            "k": args.k,
            "target_fpr": args.target_fpr,
            "defense": DEFENSE,
            "flag_rule": "score > cut (GBDT score is malicious-risk probability)",
            "cut_rule": "numpy quantile(honest_scores, 1 - target_fpr), "
                        "linear interpolation",
            "dev_signals_dir": str(args.dev_signals),
            "heldout_signals_dir": str(args.heldout_signals),
            "models_dir": str(args.models),
            "model_sha256": model_sha,
            "n_dev_cells": len(dev_cells),
            "n_heldout_cells": len(heldout_cells),
            "heldout_seeds": heldout_seeds,
            "dominance_scenarios": DOMINANCE,
            "margin": MARGIN,
            "criterion_alpha": CRITERION_ALPHA,
        },
        "dev_thresholds": frozen,
        "primary": {**primary, "per_cell": cellmap(primary_cells)},
        "secondary": {**secondary, "per_cell": cellmap(secondary_cells)},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    thresholds_doc = {
        "_meta": {
            "purpose": "H1 family operating cuts frozen on the development "
                       "leak-free corpus before the held-out read",
            "source_dir": str(args.dev_signals),
            "n_dev_cells": len(dev_cells),
            "k": args.k,
            "target_fpr": args.target_fpr,
            "cut_rule": "numpy quantile(honest_scores, 1 - target_fpr), "
                        "linear interpolation; flag = score > cut",
            "defense": DEFENSE,
            "models_dir": str(args.models),
            "model_sha256": model_sha,
        },
        "thresholds": frozen,
    }
    with open(args.out / "thresholds_dev_frozen.json", "w") as fh:
        json.dump(thresholds_doc, fh, indent=1, sort_keys=True)
    with open(args.out / "h1_corrected_read.json", "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    write_readme(args.out / "READ.md", payload)
    print(f"\n[h1_corrected_read] artifacts written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
