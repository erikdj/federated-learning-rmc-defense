"""Frozen Mann-Whitney AUC primitive used by the public adjudicator."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import revalidate_v115 as R  # noqa: E402

R.DATA = os.environ["DEV_SIG_DIR"]
SCOPE = ["krum_tge"]
A = "alie"
SCENS = ["S0", "S1", "S2", "S3", "S4"]
BUCKETS = [(1, 3, "coldstart 1-3"), (4, 10, "mid 4-10"), (11, 10 ** 9, "tenured 11+")]


def auc(mal: np.ndarray, hon: np.ndarray) -> float | None:
    """Rank-based Mann-Whitney AUC, higher score = more malicious."""
    if mal.size == 0 or hon.size == 0:
        return None
    allv = np.concatenate([mal, hon])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    srt = allv[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rm = ranks[:mal.size].sum()
    return float((rm - mal.size * (mal.size + 1) / 2) / (mal.size * hon.size))


def smd(a: np.ndarray, b: np.ndarray) -> float | None:
    """Standardized mean difference (a - b) / pooled SD."""
    if a.size < 2 or b.size < 2:
        return None
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0
