"""Szelag chain-of-custody gate (Unit D): re-confirm the published anchor on the
current code + scenario family before the confirmatory phase.

Anchor provenance: results/20260402/szelag_reproduction/ — Edge-IIoT Krum-dynamic-RMC
reproduces Szelag (2025) final_acc 0.9791 +/- 0.0037 (BRFSS analog matches 0.505 to
within 0.0012). Re-running Krum on the anchor scenario at ~5K scale under current code
and matching this number closes the chain of custody Szelag -> RMC v1.0 -> S0-S4 ->
current code, gating the full-data confirmatory phase.
"""
import argparse
import json
import sys

EDGE_ANCHOR = 0.9791
DEFAULT_TOL = 0.01


def within_tolerance(final_acc: float, anchor: float = EDGE_ANCHOR, tol: float = DEFAULT_TOL) -> bool:
    return abs(float(final_acc) - float(anchor)) <= tol


def main():
    ap = argparse.ArgumentParser(description="Szelag anchor chain-of-custody gate.")
    ap.add_argument("--result-json", required=True, help="Krum-on-anchor-scenario result JSON")
    ap.add_argument("--anchor", type=float, default=EDGE_ANCHOR)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    a = ap.parse_args()
    d = json.load(open(a.result_json))
    traj = d.get("trajectory") or []
    final_acc = traj[-1].get("accuracy") if traj else None
    ok = final_acc is not None and within_tolerance(final_acc, a.anchor, a.tol)
    print(f"[{'OK' if ok else 'FAIL'}] Szelag anchor: final_acc={final_acc} vs {a.anchor}+/-{a.tol}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
