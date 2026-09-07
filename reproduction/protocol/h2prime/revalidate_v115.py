"""Frozen H2-prime numerical primitives used by the public adjudicator."""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

DATA = os.environ.get("DEV_SIG_DIR")

DEV_SEEDS = [42, 137, 256, 314, 500]
ATTACKS = ["alie", "gaussian_noise", "label_flip"]
SCEN_SHORT = {
    "s0_clean_baseline": "S0", "s1_benign_churn_only": "S1",
    "s2_adaptive_switching_only": "S2", "s3_identity_reset_only": "S3",
    "s4_full_mix": "S4",
}
TARGET_FPR = 0.10
COLDSTART_K = 3

# v1.15 s4 / DECISION F3 frozen geometry
BAND_A_SLICE = "S4"
BAND_B_SLICES = {"alie": ["S2", "S3", "S4"],
                 "gaussian_noise": ["S2", "S3", "S4"],
                 "label_flip": ["S2", "S4"]}          # label_flip x S3 VACUOUS
CRITERION_S4_RECALL = 0.85
CRITERION_MARGIN_PP = 0.05
CRITERION_ALPHA = 0.0312

# v1.15 s2.2 feature set. Window stats are DERIVED (not logged) -- see memo s2.
LOAD_BEARING = ["update_norm", "train_loss", "num_examples",
                "norm_variance", "loss_slope"]
SECONDARY = ["cos_to_median", "L2_to_median", "cos_drift", "cos_variance"]
FEATS_V115 = LOAD_BEARING + SECONDARY
FEATS_PROTOTYPE = ["update_norm", "cos_to_median", "L2_to_median",
                   "train_loss", "num_examples"]
WINDOW = 3   # rounds; matches the v1.3 F6 cold-start window k=3

RAW = ["update_norm", "cos_to_median", "L2_to_median", "train_loss", "num_examples"]


# --------------------------------------------------------------------------
# load + derive causal window statistics
# --------------------------------------------------------------------------
def derive_window_feats(rows: list[dict]) -> None:
    """Add norm_variance / loss_slope / cos_drift / cos_variance in place.

    CAUSAL and episode-local: computed per (logical_cid, tenure-episode) over a
    trailing window of <=WINDOW rounds using PAST+CURRENT rows only, so no future
    round can leak into a row's features. A tenure reset (RMC rejoin, tenure
    restarts at 1) starts a fresh episode -- a reconnecting identity never
    inherits its previous episode's statistics.
    """
    by_cid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cid[r["logical_cid"]].append(r)
    for cid, rs in by_cid.items():
        rs.sort(key=lambda r: r["scenario_round"])
        episode: list[dict] = []
        prev_tenure = None
        for r in rs:
            t = r.get("tenure")
            if prev_tenure is not None and t is not None and t <= prev_tenure:
                episode = []            # rejoin -> new episode
            prev_tenure = t
            episode.append(r)
            w = episode[-WINDOW:]
            norms = [x["update_norm"] for x in w if x.get("update_norm") is not None]
            losses = [x["train_loss"] for x in w if x.get("train_loss") is not None]
            coss = [x["cos_to_median"] for x in w if x.get("cos_to_median") is not None]
            r["norm_variance"] = float(np.var(norms)) if len(norms) > 1 else 0.0
            r["cos_variance"] = float(np.var(coss)) if len(coss) > 1 else 0.0
            if len(losses) > 1:
                xs = np.arange(len(losses), dtype=float)
                r["loss_slope"] = float(np.polyfit(xs, np.asarray(losses), 1)[0])
            else:
                r["loss_slope"] = 0.0
            r["cos_drift"] = (float(coss[-1] - coss[-2]) if len(coss) > 1 else 0.0)


def load() -> list[dict]:
    rows: list[dict] = []
    for fn in sorted(glob.glob(f"{DATA}/*.jsonl")):
        stem = os.path.basename(fn)[: -len(".jsonl")]
        scen, defense, _exec, seedtok = stem.split("__")
        seed = int(seedtok.replace("seed", ""))
        frows = [json.loads(l) for l in open(fn) if l.strip()]
        # chain-of-custody: filename identity must match row identity
        for r in frows:
            assert r["seed"] == seed, (fn, r["seed"])
            assert SCEN_SHORT[scen] == SCEN_SHORT[r["scenario"].lower()], fn
            r["_scen"] = SCEN_SHORT[scen]
            r["_seed"] = seed
            r["_defense"] = defense
        derive_window_feats(frows)
        rows.extend(frows)
    return rows


# --------------------------------------------------------------------------
# scoring primitives (mirror analyze_variance_envelope conventions)
# --------------------------------------------------------------------------
def cut_from_calibration(honest_scores, higher_is_trust: bool) -> float:
    a = np.asarray(honest_scores, dtype=float)
    return float(np.quantile(a, TARGET_FPR if higher_is_trust else 1 - TARGET_FPR))


def flagged(scores, cut: float, higher_is_trust: bool) -> np.ndarray:
    a = np.asarray(scores, dtype=float)
    return (a < cut) if higher_is_trust else (a > cut)


def exact_wilcoxon_onesided(diffs) -> tuple[float, float]:
    """Exact one-sided signed-rank (H1: median > 0). Mirrors the H2 read."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    ranks = {}
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    total = sorted(ranks.values())
    count = 0
    allsub = 0
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            allsub += 1
            if sum(total[c] for c in combo) >= w_plus:
                count += 1
    return float(w_plus), count / allsub


def bootstrap_ci(vals, n=10000, seed=12345, alpha=0.05):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return None, None
    if a.size == 1:
        return float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n, a.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
def cell_recall(scores, mal_mask, cut, higher_is_trust):
    """One (scenario, seed) cell -> recall on the held-out family + realized FPR
    on the cell's honest rows (the FPR is REPORTED, never used to set the cut)."""
    f = flagged(scores, cut, higher_is_trust)
    mal = np.asarray(mal_mask, dtype=bool)
    rec = float(f[mal].mean()) if mal.any() else None
    fpr = float(f[~mal].mean()) if (~mal).any() else None
    return rec, fpr, int(mal.sum()), int((~mal).sum())


def run_arm(rows, *, arm, feats, defense_scope, leaky, coldstart=False):
    """arm 'leaky'      : prototype protocol (cut on the SCORED cell's honest rows)
       arm 'nested'     : v1.15 s2.2 (cut on an independent calibration SEED)

    Nested seed partition (pre-stated, deterministic rotation over sorted dev
    seeds): test = seeds[i]; calibration = seeds[(i+1) % 5]; fit = the other 3.
    Fit rows, calibration rows and scored rows are three DISJOINT seed
    partitions, so no scored-cell row -- honest or malicious -- touches either
    the model fit or the operating point.
    """
    pool = [r for r in rows if defense_scope is None or r["_defense"] in defense_scope]
    if coldstart:
        pool = [r for r in pool if (r.get("tenure") or 99) <= COLDSTART_K]
    out = {}
    for A in ATTACKS:
        per_cell = {}
        for i, ts in enumerate(DEV_SEEDS):
            cs = DEV_SEEDS[(i + 1) % len(DEV_SEEDS)]
            if leaky:
                fit_seeds = [s for s in DEV_SEEDS if s != ts]
            else:
                fit_seeds = [s for s in DEV_SEEDS if s not in (ts, cs)]
            tr = [r for r in pool
                  if r["_seed"] in fit_seeds
                  and not (r["malicious_gt"] and r.get("attack_type") == A)]
            X = np.array([[r[f] for f in feats] for r in tr], dtype=float)
            y = np.array([bool(r["malicious_gt"]) for r in tr])
            clf = GradientBoostingClassifier(random_state=0).fit(X, y)

            def pm(rs):
                if not rs:
                    return np.empty(0)
                return clf.predict_proba(
                    np.array([[r[f] for f in feats] for r in rs], dtype=float))[:, 1]

            # --- operating point, frozen BEFORE any test row is opened -------
            if leaky:
                cal_rows = None      # deferred: uses the scored cell's own honest rows
                cut = None
            else:
                cal = [r for r in pool if r["_seed"] == cs and not r["malicious_gt"]
                       and not (r["malicious_gt"] and r.get("attack_type") == A)]
                cut = cut_from_calibration(pm(cal), higher_is_trust=False)
                cal_realized = float(flagged(pm(cal), cut, False).mean())

            for scen in ["S0", "S1", "S2", "S3", "S4"]:
                te = [r for r in pool if r["_seed"] == ts and r["_scen"] == scen]
                te_h = [r for r in te if not r["malicious_gt"]]
                te_a = [r for r in te if r["malicious_gt"] and r.get("attack_type") == A]
                if not te_a or not te_h:
                    continue
                sh, sa = pm(te_h), pm(te_a)
                c = cut_from_calibration(sh, higher_is_trust=False) if leaky else cut
                rec, fpr, nm, nh = cell_recall(np.concatenate([sa, sh]),
                                               [True] * len(sa) + [False] * len(sh),
                                               c, False)
                per_cell[(scen, ts)] = {"recall": rec, "fpr": fpr,
                                        "n_mal": nm, "n_honest": nh, "cut": c,
                                        "calib_realized_fpr": None if leaky else cal_realized}
        out[A] = per_cell
    return out


def baseline_arm(rows, *, score_field, higher_is_trust, defense_scope, leaky,
                 coldstart=False):
    """Unsupervised/incumbent baseline under the IDENTICAL calibration regime."""
    pool = [r for r in rows
            if (defense_scope is None or r["_defense"] in defense_scope)
            and r.get(score_field) is not None]
    if coldstart:
        pool = [r for r in pool if (r.get("tenure") or 99) <= COLDSTART_K]
    out = {}
    for A in ATTACKS:
        per_cell = {}
        for i, ts in enumerate(DEV_SEEDS):
            cs = DEV_SEEDS[(i + 1) % len(DEV_SEEDS)]
            if not leaky:
                cal = [r[score_field] for r in pool
                       if r["_seed"] == cs and not r["malicious_gt"]]
                if not cal:
                    continue
                cut = cut_from_calibration(cal, higher_is_trust)
            for scen in ["S0", "S1", "S2", "S3", "S4"]:
                te = [r for r in pool if r["_seed"] == ts and r["_scen"] == scen]
                th = [r[score_field] for r in te if not r["malicious_gt"]]
                ta = [r[score_field] for r in te
                      if r["malicious_gt"] and r.get("attack_type") == A]
                if not ta or not th:
                    continue
                c = cut_from_calibration(th, higher_is_trust) if leaky else cut
                rec, fpr, nm, nh = cell_recall(ta + th,
                                               [True] * len(ta) + [False] * len(th),
                                               c, higher_is_trust)
                per_cell[(scen, ts)] = {"recall": rec, "fpr": fpr,
                                        "n_mal": nm, "n_honest": nh, "cut": c}
        out[A] = per_cell
    return out


# --------------------------------------------------------------------------
# reduction / adjudication
# --------------------------------------------------------------------------
def slice_units(per_cell, scens):
    """(seed -> recall) averaged over the requested scenario slice, then the
    s3.1 unit is the (scenario, seed) cell; we keep both grains."""
    cells = {k: v for k, v in per_cell.items() if k[0] in scens}
    by_seed = defaultdict(list)
    for (sc, sd), v in cells.items():
        if v["recall"] is not None:
            by_seed[sd].append(v["recall"])
    return {sd: mean(vs) for sd, vs in by_seed.items()}, cells


def summarize(per_cell, scens):
    by_seed, cells = slice_units(per_cell, scens)
    vals = [by_seed[s] for s in sorted(by_seed)]
    cellvals = [v["recall"] for v in cells.values() if v["recall"] is not None]
    fprs = [v["fpr"] for v in cells.values() if v["fpr"] is not None]
    lo, hi = bootstrap_ci(vals) if vals else (None, None)
    return {
        "scenarios": scens,
        "n_seed_units": len(vals),
        "n_cells": len(cellvals),
        "mean_recall_seedunit": mean(vals) if vals else None,
        "sd_recall_seedunit": pstdev(vals) if len(vals) > 1 else 0.0,
        "ci95_recall_seedunit": [lo, hi],
        "mean_recall_cellunit": mean(cellvals) if cellvals else None,
        "sd_recall_cellunit": pstdev(cellvals) if len(cellvals) > 1 else 0.0,
        "per_seed": {str(s): by_seed[s] for s in sorted(by_seed)},
        "mean_realized_fpr_on_scored_honest": mean(fprs) if fprs else None,
        "max_realized_fpr": max(fprs) if fprs else None,
    }


def exact_prototype_replication(rows):
    """Byte-for-byte the prototype's construction, to prove the data pipeline
    reproduces the memo's 0.792 / 0.845 / 0.996 before anything is changed:
    all defenses pooled, 5 raw features, LOSO train (4 seeds), ONE recall per
    test seed with all scenarios' rows POOLED at row level, threshold from the
    test seed's OWN honest rows (the leak)."""
    out = {}
    for A in ATTACKS:
        recs, geos, fprs = [], [], []
        for ts in DEV_SEEDS:
            tr = [r for r in rows if r["_seed"] != ts
                  and not (r["malicious_gt"] and r.get("attack_type") == A)]
            te = [r for r in rows if r["_seed"] == ts]
            te_h = [r for r in te if not r["malicious_gt"]]
            te_a = [r for r in te if r["malicious_gt"] and r.get("attack_type") == A]
            if not te_a:
                continue
            X = np.array([[r[f] for f in FEATS_PROTOTYPE] for r in tr], dtype=float)
            y = np.array([bool(r["malicious_gt"]) for r in tr])
            clf = GradientBoostingClassifier(random_state=0).fit(X, y)
            ps = clf.predict_proba(np.array([[r[f] for f in FEATS_PROTOTYPE]
                                             for r in te_h], dtype=float))[:, 1]
            pa = clf.predict_proba(np.array([[r[f] for f in FEATS_PROTOTYPE]
                                             for r in te_a], dtype=float))[:, 1]
            thr = float(np.quantile(ps, 1 - TARGET_FPR))
            recs.append(float((pa > thr).mean()))
            fprs.append(float((ps > thr).mean()))
            gh = [r["L2_to_median"] for r in te_h]
            ga = [r["L2_to_median"] for r in te_a]
            gthr = float(np.quantile(gh, 1 - TARGET_FPR))
            geos.append(float((np.asarray(ga) > gthr).mean()))
        out[A] = {"sup_mean": mean(recs), "sup_std": pstdev(recs),
                  "geo_mean": mean(geos), "geo_std": pstdev(geos),
                  "realized_fpr_mean": mean(fprs), "folds": len(recs),
                  "per_seed": dict(zip(map(str, DEV_SEEDS), recs))}
    out["MEAN"] = {"sup": mean(out[A]["sup_mean"] for A in ATTACKS),
                   "geo": mean(out[A]["geo_mean"] for A in ATTACKS)}
    return out
