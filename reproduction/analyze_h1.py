#!/usr/bin/env python3
"""Recompute the H1 held-out verdict from Krum+TGE signal logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_h2_dev_read import CRITERION_ALPHA, exact_wilcoxon_onesided
from h1_signal_family_eval import _compute_recall_at_fpr

SCENARIOS = [
    "s0_clean_baseline", "s1_benign_churn_only", "s2_adaptive_switching_only",
    "s3_identity_reset_only", "s4_full_mix",
]
SHORT = {scenario: f"S{i}" for i, scenario in enumerate(SCENARIOS)}
SEEDS = [1009, 1733, 2521, 3299, 4127, 5051, 6079, 7177, 8231, 9337]
MODEL_HASHES = {
    "S": "9638fede835cc72498e503271f5090ae00c47ae6fe2cd24d1dd01aecebc305f5",
    "W": "590e57ee788431fff704fa3724fa5f16bb984f0df364f42573fab499e75c26ae",
    "C": "9271838d7d14c513f731def94c16725a242bf012cce4bb28f3c18d02639e71a2",
}
K = 3
TARGET_FPR = 0.10
DOMINANCE_SCENARIOS = ["S2", "S3", "S4"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signal_path(root: Path, scenario: str, seed: int) -> Path:
    return root / f"{scenario}__krum_tge__persistent_optimizer__seed{seed}.jsonl"


def load_rows(root: Path, scenario: str, seed: int) -> list[dict]:
    path = signal_path(root, scenario, seed)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty signal log: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-root", required=True, type=Path)
    parser.add_argument(
        "--models-dir", type=Path,
        default=REPO / "models" / "h1_signal_family" / "leakfree_25cell",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--check-against", type=Path,
        help="fail unless the recomputed JSON object equals this reference object",
    )
    args = parser.parse_args()

    classifiers = {}
    for family in ("S", "W", "C"):
        path = args.models_dir / f"{family}_k3_final.pkl"
        actual_hash = sha256_file(path)
        if actual_hash != MODEL_HASHES[family]:
            raise SystemExit(
                f"frozen {family} model hash mismatch: expected {MODEL_HASHES[family]}, "
                f"found {actual_hash} at {path}"
            )
        with path.open("rb") as stream:
            classifiers[family] = pickle.load(stream)["model"]

    per_cell: dict[tuple[str, int], dict] = {}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            rows = load_rows(args.signals_root, scenario, seed)
            per_cell[(scenario, seed)] = {
                family: _compute_recall_at_fpr(
                    classifiers[family], rows, family, K, TARGET_FPR
                )
                for family in ("S", "W", "C")
            }

    def aggregate(scenario: str, family: str, field: str):
        values = [
            per_cell[(scenario, seed)][family][field]
            for seed in SEEDS
            if per_cell[(scenario, seed)][family][field] is not None
        ]
        return mean(values) if values else None

    margins = {}
    for scenario in SCENARIOS:
        recall_s = aggregate(scenario, "S", "recall_at_fpr")
        recall_w = aggregate(scenario, "W", "recall_at_fpr")
        recall_c = aggregate(scenario, "C", "recall_at_fpr")
        margins[SHORT[scenario]] = {
            "C": recall_c,
            "S": recall_s,
            "W": recall_w,
            "C_minus_S": recall_c - recall_s,
            "C_minus_W": recall_c - recall_w,
            "C_actual_fpr": aggregate(scenario, "C", "actual_fpr"),
            "S_actual_fpr": aggregate(scenario, "S", "actual_fpr"),
        }

    criterion_a = all(margins[s]["C_minus_S"] >= 0.05 for s in DOMINANCE_SCENARIOS)
    criterion_b = all(margins[s]["C_minus_W"] >= 0.05 for s in DOMINANCE_SCENARIOS)
    wilcoxon = {}
    for short in DOMINANCE_SCENARIOS:
        scenario = next(s for s, label in SHORT.items() if label == short)
        diffs_cs = [
            per_cell[(scenario, seed)]["C"]["recall_at_fpr"]
            - per_cell[(scenario, seed)]["S"]["recall_at_fpr"]
            for seed in SEEDS
        ]
        diffs_cw = [
            per_cell[(scenario, seed)]["C"]["recall_at_fpr"]
            - per_cell[(scenario, seed)]["W"]["recall_at_fpr"]
            for seed in SEEDS
        ]
        _, p_cs = exact_wilcoxon_onesided(diffs_cs)
        _, p_cw = exact_wilcoxon_onesided(diffs_cw)
        wilcoxon[short] = {
            "p_C_gt_S": p_cs,
            "p_C_gt_W": p_cw,
            "n_pairs": len(diffs_cs),
            "pass": p_cs <= CRITERION_ALPHA and p_cw <= CRITERION_ALPHA,
        }
    criterion_c = all(wilcoxon[s]["pass"] for s in DOMINANCE_SCENARIOS)

    output = {
        "_meta": {
            "read": "H1 confirmatory held-out; frozen detectors ce483fee; per-(scenario,seed) cell (collision-free); committed _compute_recall_at_fpr",
            "k": K,
            "target_fpr": TARGET_FPR,
            "seeds": SEEDS,
            "fpr_degeneracy_note": "realized FPR at nominal 10% reported per cell; dev stage saw ~0.88 on ~42 honest cold-start rows; per-cell honest support here ~33-42 (same regime)",
        },
        "per_scenario": margins,
        "wilcoxon": wilcoxon,
        "verdict": {
            "a_C_gt_S_5pp_all_S2S3S4": criterion_a,
            "b_C_gt_W_5pp_all_S2S3S4": criterion_b,
            "c_wilcoxon_all": criterion_c,
            "H1": (
                "H1 CONFIRMED (held-out)" if criterion_a and criterion_b and criterion_c
                else "H1 NOT MET (held-out) — falsification condition fired"
            ),
        },
    }

    if args.check_against is not None:
        reference = json.loads(args.check_against.read_text(encoding="utf-8"))
        if output != reference:
            raise SystemExit(f"recomputed H1 object differs from {args.check_against}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
