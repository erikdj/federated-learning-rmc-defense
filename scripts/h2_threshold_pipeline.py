"""F1/F8: per-defense recall@FPR threshold — select on dev honest scores, freeze, apply.

Defender convention (cf. compute_recall_fpr.py): clients with LOW score are flagged.
The threshold is the largest cutoff holding honest FPR <= target on the dev split.
"""
import argparse
import glob
import json
import numpy as np


def select_threshold(honest_scores, target_fpr: float = 0.10) -> float:
    """Largest threshold t such that fraction(honest < t) <= target_fpr."""
    s = np.sort(np.asarray(honest_scores, float))
    n = len(s)
    if n == 0:
        return float("-inf")
    idx = int(np.floor(target_fpr * n))
    return float(s[idx]) if idx < n else float(s[-1])


def freeze_thresholds(per_defense: dict, path: str) -> None:
    json.dump({"_meta": {"purpose": "frozen per-defense recall@10%FPR thresholds (F8); "
                          "selected on dev honest scores before confirmatory phase"},
               "thresholds": per_defense}, open(path, "w"), indent=2)


def load_thresholds(path: str) -> dict:
    return json.load(open(path))["thresholds"]


def _load_honest_scores(signal_glob: str, score_field: str) -> list:
    out = []
    for f in glob.glob(signal_glob):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("malicious_gt") and r.get(score_field) is not None:
                    out.append(float(r[score_field]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Select + freeze a per-defense recall@FPR threshold from dev honest scores.")
    ap.add_argument("--dev-signal-glob", required=True, help="glob of dev-seed signal logs for ONE defense")
    ap.add_argument("--score-field", required=True)
    ap.add_argument("--defense", required=True, help="defense name (dict key in the frozen file)")
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--out", required=True, help="thresholds JSON to create/update")
    a = ap.parse_args()
    honest = _load_honest_scores(a.dev_signal_glob, a.score_field)
    th = select_threshold(honest, a.target_fpr)
    try:
        existing = load_thresholds(a.out)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        existing = {}
    existing[a.defense] = th
    freeze_thresholds(existing, a.out)
    print(f"[threshold] {a.defense}: {th:.6f} (from {len(honest)} dev honest scores, FPR<={a.target_fpr})")


if __name__ == "__main__":
    main()
