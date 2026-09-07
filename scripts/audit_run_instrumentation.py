"""Per-defense instrumentation audit — pre-launch gate.

Verifies that a completed run produced EVERY metric and instrumentation channel
the downstream analysis needs, not just return_code 0. Catches the sparse,
config-specific gaps that "the run succeeded" hides — e.g. a defense that never
logs its per-client detection score, making recall@10%FPR (the H2 primary
metric) silently uncomputable for that whole arm.

Usage:
    python scripts/audit_run_instrumentation.py \
        --defense TrustScore \
        --result-json results/EXP-00X/phase4_flower__trustscore__seed42.json \
        --signal-log signals/flower_persistent__S4_full_mix__trustscore__seed42.jsonl

Exit code 0 = fully instrumented; 1 = gap found.

Context:
    confounder_control is EXPECTED absent on homogeneous scenarios (S0-S4); it is
    only emitted for the heterogeneous v3 confounder arm (runner attaches it under
    `if cc:`). The trajectory persists round/f1/accuracy/loss by contract;
    precision/recall are computed in fit-progress but not persisted and are not
    needed for H1-H4 (H4 uses macro-F1 = f1; recall@FPR comes from signal scores).
"""
import argparse
import json
import sys

SCORE_FIELD = {"krum": "krum_score", "trustscore": "trust_score",
               "tgensemble": "tge_score", "tge": "tge_score",
               "krum+tge": "tge_score"}  # chained: TGE plugin is the scoring layer
REQUIRED_RESULT_BLOCKS = ["trajectory", "convergence", "defense_overhead"]
REQUIRED_TRAJ_KEYS = ["accuracy", "f1", "loss"]  # the persisted trajectory contract


def normalize_defense(defense: str) -> str:
    return defense.replace("Scenario", "").lower()


def audit(defense: str, result_json: str, signal_log: str) -> list[str]:
    gaps: list[str] = []
    token = normalize_defense(defense)
    score_field = SCORE_FIELD.get(token)
    if score_field is None:
        gaps.append(f"unknown defense token {token!r} (no score field mapping)")
        return gaps

    # ---- result JSON ----
    try:
        d = json.load(open(result_json))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return [f"result JSON unreadable: {e}"]
    if d.get("return_code") != 0:
        gaps.append(f"return_code != 0 ({d.get('return_code')})")
    traj = d.get("trajectory") or []
    if not traj:
        gaps.append("trajectory empty")
    elif not all(isinstance(traj[-1].get(k), (int, float)) for k in REQUIRED_TRAJ_KEYS):
        missing = [k for k in REQUIRED_TRAJ_KEYS if not isinstance(traj[-1].get(k), (int, float))]
        gaps.append(f"trajectory missing persisted keys: {missing}")
    for b in REQUIRED_RESULT_BLOCKS:
        v = d.get(b)
        if v is None or (hasattr(v, "__len__") and len(v) == 0):
            gaps.append(f"result block missing/empty: {b}")

    # ---- signal log ----
    try:
        rows = [json.loads(l) for l in open(signal_log) if l.strip()]
    except FileNotFoundError:
        return gaps + [f"signal log missing: {signal_log}"]
    if not rows:
        return gaps + ["signal log has no rows"]
    if sum(1 for r in rows if r.get("malicious_gt")) == 0:
        gaps.append("malicious_gt never True (recall@FPR has no positive class)")
    scored = [r for r in rows if r.get(score_field) is not None]
    if not scored:
        gaps.append(f"CRITICAL: {score_field} is None for ALL rows -> recall@10%FPR "
                    f"UNCOMPUTABLE for {defense}")
    else:
        mal = sum(1 for r in scored if r.get("malicious_gt"))
        hon = sum(1 for r in scored if not r.get("malicious_gt"))
        if mal == 0 or hon == 0:
            gaps.append(f"recall@FPR not computable: scored malicious={mal} honest={hon}")
    for meta in ("logical_cid", "server_round" if rows[0].get("server_round") is not None else "scenario_round"):
        if all(r.get(meta) is None for r in rows):
            gaps.append(f"metadata field absent: {meta}")
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True)
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--signal-log", required=True)
    args = ap.parse_args()
    gaps = audit(args.defense, args.result_json, args.signal_log)
    if gaps:
        print(f"[FAIL] {args.defense}: {len(gaps)} gap(s)")
        for g in gaps:
            print(f"  - {g}")
        sys.exit(1)
    print(f"[OK] {args.defense}: fully instrumented (trajectory, metric blocks, "
          f"per-client {SCORE_FIELD[normalize_defense(args.defense)]}, malicious_gt, metadata).")
    sys.exit(0)


if __name__ == "__main__":
    main()
