"""Shared foundation for the H2′ confirmatory read.

Frozen constants transcribed from the ratified pre-registration, the corpus
profiles, the two refusal classes, the imports of the frozen dev harness, and
the statistical primitives the bands are decided by. Everything here is either
transcribed from the spec or imported from the committed harness — nothing in
this module is a modeling choice.

Split out of `adjudicate_h2prime.py` (2026-08-12) purely for file size; no
behaviour changed, which is verified by the dev-smoke output remaining
byte-identical across the split.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO / "reproduction" / "protocol" / "h2prime"
BUILDER_PATH = HARNESS_DIR / "revalidate_v115.py"
BLENDED_PATH = HARNESS_DIR / "blended_loao.py"
GOLDEN_TEST_PATH = REPO / "tests" / "test_window_feats_golden.py"
SEALED_SEED_PATH = REPO / "data" / "h2prime_confirm_seeds.json"
# EXP-048's seed authority. Long unsealed — EXP-048 was read and published on
# 2026-08-09 — but it is still the REGISTERED universe for the § 4 secondary 9
# contrast, so the expected grid is derived from this file and never from the
# staged files the gate is meant to be checking.
H2_CONFIRM_SEED_PATH = REPO / "data" / "h2_confirm_seeds.json"
H2_CONFIRM_SEED_SHA256 = (
    "19e8eac72025dbd7b3adc05c35ea3fd871518ba3be6f53d7953f5ef8c44be19b")
H2_CONFIRM_SEED_KEY = "h2_confirm_seeds"

# --------------------------------------------------------------------------
# frozen constants — transcribed from the ratified pre-registration
# --------------------------------------------------------------------------
SEALED_SEED_SHA256 = "db97f8d38232f6eacf717c413ccb0aa9d0b2c1ffa0d9632c0351a30470df6d04"
SCENARIOS = ["S0", "S1", "S2", "S3", "S4"]          # § 4.1 frozen cell table
P1_SLICE = "S4"                                     # § 4 (P1) the switching surface
P1_FLOOR = 0.35                                     # § 4 (P1)
P2_POPULATION_FAMILY = "alie"                       # § 4 (P2)
COMPARABILITY_INTERVAL = (0.08, 0.12)               # § 3.2, closed interval
ALPHA = 0.05                                        # § 4 director ruling 2026-08-10
GBDT_PARAMS = dict(                                 # § 2.1a, enumerated verbatim
    loss="log_loss",
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    init=None,
    random_state=0,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0,
)
PINNED_SKLEARN = "1.7.2"                            # § 2.1a
PINNED_NUMPY = "2.2.6"                              # § 2.2a item 7
DEV_SEEDS = [42, 137, 256, 314, 500]                # EXP-011 + EXP-012/013 refills
# The REGISTERED configuration token (EXP-051 § 5 item 5: unit ids are
# `S*__krum_tge__persistent_optimizer__seed<sealed>`, signal-log defense class
# `krumtge`). Homogeneity alone is not enough — a uniformly WRONG map, e.g. 50
# standalone-`tge` units, is homogeneous and would permanently adjudicate the
# sealed cohort on a configuration the protocol never registered.
REGISTERED_DEFENSE_TOKEN = "krum_tge"


class Refusal(RuntimeError):
    """A pre-condition of the adjudication is not met. Nothing is scored."""


class HardStop(RuntimeError):
    """A frozen rule fired mid-run (§ 2.1a item 5 / § 2.2b). Nothing is reported."""


@dataclass(frozen=True)
class Profile:
    """Corpus shape. CONFIRMATORY is the only adjudicating one."""

    name: str
    n_seeds: int
    n_cells: int
    df: int
    t_crit: float
    p2_min_positive: int
    adjudicating: bool
    # Whether every assembly-map entry MUST carry a sha256 content digest.
    # EXEMPTION, STATED NOT SILENT: the digest protects the ONE-SHOT sealed
    # read — the sealed corpus is opened exactly once, so "the path resolved"
    # is not evidence the bytes are the registered ones, and a hand-written or
    # dry-run map with the field omitted would rest custody on paths alone.
    # DEV-SMOKE runs on already-disclosed dev data, adjudicates nothing, and
    # may be re-run freely, so a digest-less dev map is a convenience with no
    # custody consequence. A digest that IS present is verified on EVERY
    # profile regardless — this flag governs the REQUIREMENT, never the check.
    requires_content_digest: bool


# § 4 (P1) CI: df = n − 1 = 9, t_{0.975,9} = 2.262. § 4 (P2): ≥ 9 of 10.
CONFIRMATORY = Profile("CONFIRMATORY", 10, 50, 9, 2.262, 9, True, True)
# Executor self-test only. n = 5 constants are the arithmetic analogues; they
# adjudicate nothing and are labelled DEV-SMOKE everywhere they appear.
DEV_SMOKE = Profile("DEV-SMOKE", 5, 25, 4, 2.776, 4, False, False)

# --------------------------------------------------------------------------
# frozen harness import (§ 2.1a / § 2.2b — imported, never re-derived)
# --------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses/typing resolve via sys.modules
    spec.loader.exec_module(module)
    return module


if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# v1.15b § 3 — the strict-identity key is the CANONICAL BASE IDENTITY, imported
# from the emitter rather than re-derived: `client_5_new2` → `client_5`, while
# the legacy `client_9_new` / `client_19_new` aliases are left alone because
# they map to partitions 10 / 20 — different devices, not RMC cycles of one.
from flowerfl.signal_logger import canonical_device_id  # noqa: E402

R = _load_module(BUILDER_PATH, "revalidate_v115_frozen")
# blended_loao.py binds R.DATA from DEV_SIG_DIR at import time; the adjudicator
# never globs a directory (it loads by assembly map), so the value is inert.
os.environ.setdefault("DEV_SIG_DIR", "")
B = _load_module(BLENDED_PATH, "blended_loao_frozen")
GOLDEN = _load_module(GOLDEN_TEST_PATH, "window_feats_golden_gate")
# § 4 secondary 4 — the Mann-Whitney AUC with the committed tie handling that
# produced the dev 0.82–0.92 reference, imported rather than re-implemented.
AUC = _load_module(HARNESS_DIR / "s4_alie_diagnostic.py", "s4_alie_diagnostic_frozen").auc

BASELINES = B.BASELINES              # § 3.1 the three enumerated per-client scores

# The INPUT CONTRACT of the scoring path, assembled from the frozen builder's
# own constants rather than restated from memory, so a change there propagates
# here instead of drifting:
#   * `R.RAW` — the raw per-row features `design_matrix` reads (the derived
#     window features are produced by the builder, never required on input);
#   * the structural keys `derive_window_feats` itself consumes: `logical_cid`
#     (grouping), `scenario_round` (ordering) and `tenure` (episode boundary);
#   * the label and identity keys the loader and scorer assert on.
# A pre-read gate that checks fewer keys than this lets a row pass validation
# and then throw inside the single sealed pass, which cannot be retried.
SCORING_STRUCTURAL_KEYS = ("logical_cid", "scenario_round", "tenure")
SCORING_LABEL_KEYS = ("malicious_gt", "attack_type")
SCORING_IDENTITY_KEYS = ("seed", "scenario")
# The § 3.1 BASELINE INSTRUMENTS, taken from the registered roster rather than
# retyped. `design_matrix` is not the only consumer of a row: the P2 oracle-max
# arm reads each enumerated instrument directly off the row. Two of the three
# (L2_to_median, cos_to_median) are also R.RAW features and were already
# covered by accident of that overlap; `krum_score` is a baseline instrument
# ONLY, and was covered by neither contract.
#
# Presence is REQUIRED, not merely typed, because absence is silent here: the
# arm reads `r[b] for r in ... if r.get(b) is not None`, so a missing column
# yields an empty score vector and `continue`s, dropping that instrument from
# the oracle maximum. v1.15b § 1.2 item 3 freezes the P2 tie-break to the FIRST
# entry of this tuple, so losing `krum_score` would silently move the guarded
# arg-max identity to L2_to_median under the exact-0.000 ties § 4 expects as the
# normal regime — a spec-level change of meaning with no error anywhere.
SCORING_BASELINE_KEYS = tuple(b for b, _trust, _label in BASELINES)
SCORING_INPUT_KEYS = tuple(sorted(
    set(R.RAW) | set(SCORING_STRUCTURAL_KEYS) | set(SCORING_LABEL_KEYS)
    | set(SCORING_IDENTITY_KEYS) | set(SCORING_BASELINE_KEYS)))
# The EXP-048 arm additionally feeds the TGE side of the matched contrast, and
# `tge_score` is read with `.get()` — a missing key would silently shrink
# coverage instead of failing, which is precisely the class of defect the
# pre-read gate exists to catch. It is required on the standalone-TGE arm,
# whose defining property is that every row reaches the TGE stage.
EXP048_REQUIRED_KEYS = tuple(sorted(set(SCORING_INPUT_KEYS) | {"tge_score"}))

# The VALUE contract, from the same single source as the key contract. Key
# presence is not enough: a null or wrongly-typed value passes a presence check
# and then fails INSIDE the sealed pass, which cannot be retried. Each consumed
# key is classified by how the scoring path actually uses it.
#   "numeric"          — read as a float into the design matrix or a cut
#   "numeric_or_null"  — read as a number when present; null is MEANINGFUL
#   "int"              — used as an ordering key or an identity
#   "int_or_null"      — `tenure`: compared for the episode boundary, may be null
#   "bool"             — the label
#   "str"              — identity / grouping / family label
SCORING_VALUE_CONTRACT = {
    **{k: "numeric" for k in R.RAW},
    "scenario_round": "int",          # derive_window_feats sort key
    "tenure": "int_or_null",          # episode boundary, null-tolerant by design
    "seed": "int",                    # loader identity assertion
    "malicious_gt": "bool",           # the label
    "attack_type": "str",             # family label ("" for discovery rows)
    "logical_cid": "str",             # window grouping key
    "scenario": "str",                # loader identity assertion
    # `tge_score` null is a MEASURED COVERAGE FACT the machinery reports, not a
    # defect; when present it is read as a number, so it is checked as one.
    "tge_score": "numeric_or_null",
    # BASELINE-ONLY instruments (`krum_score`). The same "null is a measured
    # fact" reading as `tge_score`, and here the deployed emitter states it
    # outright: across the 98,100-row EXP-011 dev corpus the key is present on
    # every row of every arm, non-null on 100% of the two Krum-bearing arms
    # (krum, krum_tge) and null on 100% of the two that never run a Krum stage
    # (tge, trustscore). The null therefore RECORDS which defense produced the
    # row; it is not a defect to reject. The baseline arm agrees — it drops
    # nulls per row and carries the counts (cal/honest/mal_null_dropped) into
    # the report so a subset can never be scored silently — and reads the
    # survivors as floats, so they are checked as numbers.
    #
    # R.RAW members are EXCLUDED here rather than overridden: L2_to_median and
    # cos_to_median serve double duty as design-matrix features, where null has
    # no tolerance at all (it becomes NaN and trips the § 2.1a hard stop inside
    # the sealed pass). Two consumers, two tolerances — the STRICTER one governs.
    **{k: "numeric_or_null" for k in SCORING_BASELINE_KEYS if k not in R.RAW},
}
# ENUM contract, from the SAME registered roster the scorer partitions on. An
# unregistered family string is not merely odd: score_corpus partitions rows by
# `attack_type`, so a stray value would create a phantom family slice that no
# LOAO fold was ever fit for. The empty string stays legal — it is the frozen
# loader's discovery-row marker (§ 3.2 row-eligibility rule).
SCORING_ENUM_CONTRACT = {"attack_type": ("",) + tuple(R.ATTACKS)}
ATTACKS = R.ATTACKS                  # alie / gaussian_noise / label_flip
FEATS = R.FEATS_V115                 # § 2.2b frozen 9-column order
TARGET_FPR = R.TARGET_FPR            # 0.10


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def exact_sign_p(diffs: list[float]) -> dict:
    """§ 4 (P2) frozen convention: zeros carry no sign and are EXCLUDED.

    p = Σ_{k=P}^{n_eff} C(n_eff, k) / 2^{n_eff} under p₀ = 0.5.
    """
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    zero = sum(1 for d in diffs if d == 0)
    n_eff = pos + neg
    if n_eff == 0:
        p = 1.0
    else:
        p = sum(math.comb(n_eff, k) for k in range(pos, n_eff + 1)) / (2 ** n_eff)
    return {"positive": pos, "negative": neg, "zero": zero, "n_eff": n_eff,
            "p_one_sided_exact": p}


def student_t_ci(vals: list[float], t_crit: float, df: int) -> dict:
    """§ 4 (P1) frozen CI: mean ± t·SD/√n, sample SD (ddof = 1), NOT clipped."""
    n = len(vals)
    if n < 2:
        return {"n": n, "mean": (vals[0] if vals else None), "sd_sample": None,
                "se": None, "t_crit": t_crit, "df": df, "lo": None, "hi": None}
    m = mean(vals)
    sd = stdev(vals)
    se = sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd_sample": sd, "se": se, "t_crit": t_crit,
            "df": df, "lo": m - t_crit * se, "hi": m + t_crit * se}


def comparable(fpr: float | None) -> bool:
    """§ 3.2 comparability interval [0.08, 0.12], CLOSED."""
    lo, hi = COMPARABILITY_INTERVAL
    return fpr is not None and lo <= fpr <= hi



# Two-sided 95 % Student-t critical values t_{0.975, df}, standard table to 3
# decimals. Frozen as constants because the reported CIs are computed on the
# RETAINED fold count, whose df is not known until the coverage floor has been
# applied — so the profile's own df/t pair (df = 9, t = 2.262) cannot be used
# against a shorter difference vector. That published pair appears at df = 9
# below, and the § 4.1 dev-profile value 2.776 at df = 4, which is the
# consistency check on this table.
T_CRIT_95_TWO_SIDED = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def ci_on_retained(vals: list[float], ci_fn) -> dict:
    """A CI whose df comes from the RETAINED sample, never from the profile.

    Fewer than two retained values leaves the interval undefined; that is
    stated in the output rather than papered over with a wider t.
    """
    n = len(vals)
    if n < 2:
        return {"status": "NOT COMPUTED", "n_retained": n,
                "reason": "fewer than 2 retained folds; no interval is defined"}
    df = n - 1
    t = T_CRIT_95_TWO_SIDED.get(df)
    if t is None:
        return {"status": "NOT COMPUTED", "n_retained": n, "df": df,
                "reason": f"df = {df} is outside the frozen t-table (1–20)"}
    return {"status": "COMPUTED", "n_retained": n, **ci_fn(vals, t, df)}


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"

