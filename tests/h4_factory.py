"""Synthetic-unit fixture factory for the frozen H4 scorer's tests.

Builds unit result JSONs shaped exactly like the Lane-B emitters on
`h4/online-serving` (`scripts/run_phase4_flower.py` result schema +
`flowerfl/h4_diagnostics.py` block), with custody fields auto-correct per
arm so each test mutates exactly the field it is about.

SEED DISCIPLINE: `TEST_SEEDS` are distinctive synthetic values (90001..)
that can never collide with sealed seeds, and several tests grep raw JSON
output for them to prove the scorer never prints a seed value anywhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from flowerfl.fingerprint_registry import (
    CALIBRATION_ARTIFACT_SHA256,
    CalibrationCohort,
    locked_metric,
    locked_tau,
)
from scripts import h4_scoring_lib as lib

#: config label -> unit-id token per the Lane-B rule label.replace('+','_').lower()
CONFIG_LABEL_BY_ARM: Dict[str, str] = {
    "h2p_fp_krum": "H2P+FP+Krum",
    "krum": "Krum",
    "h2p_fp": "H2P+FP",
    "krum_tge_fp": "Krum+TGE+FP",
    "h2p_krum": "H2P+Krum",
    "h2p_fp_ts": "H2P+FP+TS",
    "trustscore": "TrustScore",
    "fedavg": "FedAvg",
    "h2p_ts": "H2P+TS",
}

SCENARIO_FILE_BY_CODE: Dict[str, str] = {
    "C0": "C0_clean_no_attack.json",
    "S0": "S0_clean_baseline.json",
    "S1": "S1_benign_churn.json",
    "S2": "S2_adaptive_switching.json",
    "S3": "S3_identity_reset.json",
    "S4": "S4_combined.json",
}

#: Distinctive synthetic seeds — NEVER the sealed values; redaction tests
#: assert these strings are absent from every scorer output.
TEST_SEEDS: Sequence[int] = tuple(range(90001, 90011))

_DETECTOR_NAME_BY_ARM: Dict[str, Optional[str]] = {
    "h2p_fp_krum": "H2PrimeDetector",
    "h2p_fp": "H2PrimeDetector",
    "h2p_krum": "H2PrimeDetector",
    "h2p_fp_ts": "H2PrimeDetector",
    "h2p_ts": "H2PrimeDetector",
    "krum_tge_fp": "TGEnsemble",
    "krum": None,
    "trustscore": None,
    "fedavg": None,
}
_FP_NAME_BY_ARM: Dict[str, Optional[str]] = {
    arm: ("Fingerprint" if arm in lib.FP_ARMS else None)
    for arm in CONFIG_LABEL_BY_ARM
}
_AGG_NAME_BY_ARM: Dict[str, Optional[str]] = {
    "h2p_fp_krum": "KrumDefense",
    "krum": "KrumDefense",
    "h2p_fp": None,  # detect -> identity -> FedAvg
    "krum_tge_fp": "KrumDefense",
    "h2p_krum": "KrumDefense",
    "h2p_fp_ts": "TrustScore",
    "trustscore": "TrustScore",
    "fedavg": None,
    "h2p_ts": "TrustScore",
}


#: Deterministic fake bundle_v2 sha for scorer tests. Erratum B (v1.53) made
#: the shipped pin a TBD_BUNDLE_V2 sentinel until the real bundle v2 is built,
#: so every test that exercises the scoring pipeline pins THIS value first
#: (via `pin_fake_bundle_v2`) — mirroring the post-calibration pinned state.
#: Sentinel-refusal behavior itself is covered by tests/test_h4_scorer_pin_v2.py.
FAKE_BUNDLE_V2_SHA = "f2" * 32


def pin_fake_bundle_v2(monkeypatch) -> str:
    """Swap the sentinel pin for the deterministic fake bundle_v2 sha."""
    monkeypatch.setattr(lib, "SERVING_BUNDLE_SHA256", FAKE_BUNDLE_V2_SHA)
    return FAKE_BUNDLE_V2_SHA


def install_split_manifest(tmp_path: Path, monkeypatch,
                           content: bytes = b'{"val_indices": [1, 2], "test_indices": [3, 4]}\n') -> str:
    """Point the scorer at a temp committed-split manifest; return its sha256."""
    path = tmp_path / "val_test_split_manifest.json"
    path.write_bytes(content)
    monkeypatch.setattr(lib, "SPLIT_MANIFEST_PATH", path)
    return hashlib.sha256(content).hexdigest()


def install_seed_manifest(tmp_path: Path, monkeypatch,
                          seeds: Sequence[int] = TEST_SEEDS,
                          key: str = "confirmatory_seeds") -> Path:
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({key: list(seeds)}))
    monkeypatch.setattr(lib, "SEED_MANIFEST_PATH", path)
    return path


def make_trajectory(n_rounds: int = 8, base: float = 0.5,
                    final5: Any = 0.9, with_f1: bool = True) -> List[Dict[str, Any]]:
    """Rounds 1..n; the last five rounds carry `final5` (scalar or 5-seq)."""
    if not isinstance(final5, (list, tuple)):
        final5 = [float(final5)] * 5
    assert len(final5) == 5
    entries = []
    for r in range(1, n_rounds + 1):
        idx = r - (n_rounds - 5) - 1
        acc = float(final5[idx]) if idx >= 0 else float(base)
        entry = {"round": r, "accuracy": acc, "loss": 0.5}
        if with_f1:
            entry["f1"] = max(0.0, acc - 0.05)
        entries.append(entry)
    return entries


def make_diagnostics(arm: str, n_rounds: int = 4, kept_size: int = 10,
                     malicious_frac: float = 0.2,
                     empty_rounds: Sequence[int] = (),
                     det_honest: int = 1, det_mal: int = 2,
                     fp_dropped: int = 1, agg_rejected: int = 1) -> Dict[str, Any]:
    """A per-unit `h4_diagnostics` block matching build_h4_diagnostics output."""
    detector = _DETECTOR_NAME_BY_ARM[arm]
    fp = _FP_NAME_BY_ARM[arm]
    aggregator = _AGG_NAME_BY_ARM[arm]
    server_rounds = list(range(1, n_rounds + 1))
    kept_sizes, fracs, empties = [], [], []
    for r in server_rounds:
        if r in empty_rounds:
            kept_sizes.append(0)
            fracs.append(None)  # 0/0 — null, never zero
            empties.append(r)
        else:
            kept_sizes.append(kept_size)
            fracs.append(malicious_frac)
    per_round: Dict[str, Optional[List[int]]] = {
        "detector_dropped_honest": [det_honest] * n_rounds if detector else None,
        "detector_dropped_malicious": [det_mal] * n_rounds if detector else None,
        "fp_hard_dropped": [fp_dropped] * n_rounds if fp else None,
        "aggregator_rejected": [agg_rejected] * n_rounds if aggregator else None,
    }
    totals = {
        key: (sum(series) if series is not None else None)
        for key, series in per_round.items()
    }
    return {
        "layers_present": {"detector": detector, "fp": fp, "aggregator": aggregator},
        "rounds": {"server": server_rounds,
                   "scenario": [r - 1 for r in server_rounds]},
        "kept_set_size": kept_sizes,
        "kept_set_malicious_fraction": fracs,
        "empty_aggregate_rounds": {
            "count": len(empties),
            "server_rounds": empties,
            "scenario_rounds": [r - 1 for r in empties],
        },
        "per_layer_removals_per_round": per_round,
        "per_layer_removal_totals": totals,
    }


def make_unit(arm: str, scenario: str, seed: int, *,
              acc_final5: Any = 0.9, base_acc: float = 0.5, n_rounds: int = 8,
              trajectory: Optional[List[Dict[str, Any]]] = None,
              diagnostics: Any = "auto",
              run_uid: Optional[str] = None,
              with_f1: bool = True) -> Dict[str, Any]:
    """One synthetic unit result JSON payload with custody correct per arm."""
    if trajectory is None:
        trajectory = make_trajectory(n_rounds=n_rounds, base=base_acc,
                                     final5=acc_final5, with_f1=with_f1)
    uid = run_uid or f"uid-{arm}-{scenario}-{seed}"
    unit: Dict[str, Any] = {
        "config": CONFIG_LABEL_BY_ARM[arm],
        "strategy": CONFIG_LABEL_BY_ARM[arm],
        "seed": seed,
        "return_code": 0,
        "trajectory": trajectory,
        "final_accuracy": trajectory[-1]["accuracy"] if trajectory else None,
        "final_f1": trajectory[-1].get("f1") if trajectory else None,
        "provenance": {
            # universal run identity (director ruling 2026-08-17): ALL arms
            "run_uid": uid,
            "scenario_path": f"rmc/scenarios/{SCENARIO_FILE_BY_CODE[scenario]}",
            "eval_split": "sealed_test",
            "eval_split_manifest_sha256": lib.split_manifest_sha256(),
            "serving_bundle_sha256": (
                lib.SERVING_BUNDLE_SHA256 if arm in lib.DETECTOR_ARMS else None
            ),
            "fp_registry_policy": (
                "flag_gated" if arm in lib.FP_ARMS else None
            ),
            "fp_cohort": "validation" if arm in lib.FP_ARMS else None,
        },
    }
    if diagnostics == "auto":
        unit["h4_diagnostics"] = make_diagnostics(arm)
    elif diagnostics is not None:
        unit["h4_diagnostics"] = diagnostics
    if arm in lib.FP_ARMS:
        cohort = CalibrationCohort.VALIDATION
        unit["fingerprint_registry"] = {
            "registry_policy": "flag_gated",
            # the registry-block copy MUST equal provenance.run_uid
            "run_uid": uid,
            "tau": locked_tau(cohort),
            "metric_provenance": locked_metric(cohort).provenance,
            "calibration_artifact_sha256": CALIBRATION_ARTIFACT_SHA256,
        }
    return unit


def mutate(unit: Dict[str, Any], path: Sequence[str], value: Any,
           delete: bool = False) -> Dict[str, Any]:
    """Immutable helper: a deep copy of `unit` with one nested field changed."""
    out = copy.deepcopy(unit)
    node = out
    for key in path[:-1]:
        node = node[key]
    if delete:
        node.pop(path[-1], None)
    else:
        node[path[-1]] = value
    return out


def write_unit(dirpath: Path, unit: Dict[str, Any],
               name: Optional[str] = None) -> Path:
    """Write a unit file. The DEFAULT filename embeds the seed on purpose,
    mirroring launch-tooling naming, so redaction tests exercise the
    path-echo hazard."""
    if name is None:
        arm = str(unit.get("config", "noconfig")).replace("+", "_").lower()
        stem = Path(
            str((unit.get("provenance") or {}).get("scenario_path", "noscen"))
        ).stem
        name = f"{arm}__{stem}__seed{unit.get('seed', 'noseed')}.json"
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    path.write_text(json.dumps(unit))
    return path


def write_corpus(dirpath: Path,
                 arms: Sequence[str] = tuple(CONFIG_LABEL_BY_ARM),
                 scenarios: Sequence[str] = tuple(SCENARIO_FILE_BY_CODE),
                 seeds: Sequence[int] = TEST_SEEDS,
                 acc_fn: Optional[Callable[[str, str, int], Any]] = None,
                 unit_fn: Optional[Callable[..., Dict[str, Any]]] = None,
                 ) -> List[Path]:
    """A full (arms x scenarios x seeds) corpus on disk."""
    make = unit_fn or make_unit
    paths = []
    for arm in arms:
        for scenario in scenarios:
            for seed in seeds:
                acc = acc_fn(arm, scenario, seed) if acc_fn else 0.9
                paths.append(write_unit(
                    dirpath, make(arm, scenario, seed, acc_final5=acc)))
    return paths


def confirmed_acc(arm: str, scenario: str, seed: int) -> float:
    """A ceiling corpus: reduction +0.20 on every S1-S4 pair -> CONFIRMED."""
    if scenario == "C0":
        return 0.95
    if scenario == "S0":
        return 0.90
    if arm == "krum":
        return 0.70   # degradation 0.25 vs C0
    if arm == "h2p_fp_krum":
        return 0.90   # degradation 0.05 vs C0 -> reduction 0.20
    return 0.80


def falsified_acc(arm: str, scenario: str, seed: int) -> float:
    """S2 inverted (treatment worse) -> the conjunctive gate FALSIFIES."""
    if scenario == "S2":
        if arm == "krum":
            return 0.90   # degradation 0.05
        if arm == "h2p_fp_krum":
            return 0.70   # degradation 0.25 -> reduction -0.20
        return 0.80
    return confirmed_acc(arm, scenario, seed)
