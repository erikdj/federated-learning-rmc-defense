"""Frozen H2-prime baseline roster used by the public adjudicator."""
from __future__ import annotations

import json
import os
import sys
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import revalidate_v115 as R  # noqa: E402

R.DATA = os.environ["DEV_SIG_DIR"]

SCENS = ["S0", "S1", "S2", "S3", "S4"]
ADJUDICATING = "S4"
SUPPORTING = ["S2", "S3"]
ALPHA = 0.05                    # Erik ruling; n=5 exact min p = 1/32 = 0.03125 <= 0.05
BASELINES = [
    ("krum_score", True, "Krum (krum_score, TRUST)"),
    ("L2_to_median", False, "geometric distance-to-median (L2_to_median)"),
    ("cos_to_median", True, "cosine-to-median (TRUST-direction similarity)"),
]
SCOPE = ["krum_tge"]            # the s5 confirmatory configuration
