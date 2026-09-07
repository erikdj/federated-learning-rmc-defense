"""Erratum-B runner plumbing: observe/cuts-version CLI, custody export, the
`h2p_observe` calibration block, and the observe cache-reuse guard.

`docs/reproduction/experiments.md` (RULED,
methodology v1.53). The calibration log rides the unit's RESULT JSON as an
`h2p_observe` block (same upload path as every other result field — no new
sidecar custody), ground-truth-enriched at export from the strategy's
scenario adversary set (the SAME source `_maybe_log_signals` stamps
`malicious_gt` from).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# CLI -> run-config
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h2p_cli_args_defaults_produce_empty_overrides():
    from run_phase4_flower import add_h2p_cli_args, h2p_run_config_from_cli

    p = argparse.ArgumentParser()
    add_h2p_cli_args(p)
    args = p.parse_args([])
    assert h2p_run_config_from_cli(args.h2p_observe_only,
                                   args.h2p_cuts_version) == {}


@pytest.mark.unit
def test_h2p_cli_args_flags_map_to_run_config_keys():
    from run_phase4_flower import add_h2p_cli_args, h2p_run_config_from_cli

    p = argparse.ArgumentParser()
    add_h2p_cli_args(p)
    args = p.parse_args(["--h2p-observe-only", "--h2p-cuts-version", "v2"])
    assert h2p_run_config_from_cli(args.h2p_observe_only,
                                   args.h2p_cuts_version) == {
        "h2p-observe-only": True,
        "h2p-cuts-version": "v2",
    }


@pytest.mark.unit
def test_h2p_cuts_version_cli_is_a_closed_choice():
    from run_phase4_flower import add_h2p_cli_args

    p = argparse.ArgumentParser()
    add_h2p_cli_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--h2p-cuts-version", "v9"])


# ---------------------------------------------------------------------------
# provenance (always-declared, single-source coercion)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_h2p_observe_provenance_always_declares():
    from run_phase4_flower import h2p_observe_provenance

    assert h2p_observe_provenance({}) == {"h2p_observe_only": False}
    assert h2p_observe_provenance({"h2p-observe-only": True}) == {
        "h2p_observe_only": True}
    assert h2p_observe_provenance({"h2p-observe-only": "1"}) == {
        "h2p_observe_only": True}


@pytest.mark.unit
def test_h2p_observe_provenance_refuses_garbage():
    from run_phase4_flower import h2p_observe_provenance

    with pytest.raises(ValueError, match="h2p-observe-only"):
        h2p_observe_provenance({"h2p-observe-only": "ture"})


class _FakeDetector:
    name = "H2PrimeDetector"

    def __init__(self, observe_only=True, rows=(), arm_class="krum_family",
                 cuts_version="v1", cut=0.5, scenario_token="S3"):
        self.observe_only = observe_only
        self.bundle_sha256 = "ab" * 32
        self.arm_class = arm_class
        self.cuts_version = cuts_version
        self.cut = cut
        self.scenario_token = scenario_token
        self._rows = [dict(r) for r in rows]

    @property
    def observe_rows(self):
        return [dict(r) for r in self._rows]


def _strategy(plugins, adv_ids=frozenset()):
    return SimpleNamespace(_plugins=list(plugins), _adv_ids=set(adv_ids))


@pytest.mark.unit
def test_h4_serving_provenance_exports_cuts_version_and_arm_class():
    from run_phase4_flower import h4_serving_provenance

    det = _FakeDetector(cuts_version="v2", arm_class="ts_family")
    prov = h4_serving_provenance(_strategy([det]))
    assert prov == {
        "serving_bundle_sha256": "ab" * 32,
        "h2p_cuts_version": "v2",
        "h2p_arm_class": "ts_family",
    }


@pytest.mark.unit
def test_h4_serving_provenance_nulls_for_non_detector_arms():
    from run_phase4_flower import h4_serving_provenance

    prov = h4_serving_provenance(_strategy([]))
    assert prov == {
        "serving_bundle_sha256": None,
        "h2p_cuts_version": None,
        "h2p_arm_class": None,
    }


# ---------------------------------------------------------------------------
# the h2p_observe result-JSON block
# ---------------------------------------------------------------------------

_ROWS = [
    {"server_round": 2, "scenario_round": 1, "logical_cid": "client_0",
     "score": 0.4, "would_flag": False},
    {"server_round": 2, "scenario_round": 1, "logical_cid": "client_7",
     "score": 0.9, "would_flag": True},
]


@pytest.mark.unit
def test_h2p_observe_block_enriches_ground_truth_and_carries_meta():
    from run_phase4_flower import h2p_observe_block

    det = _FakeDetector(rows=_ROWS)
    block = h2p_observe_block(_strategy([det], adv_ids={"client_7"}))
    assert block["mode"] == "observe_only"
    assert block["cuts_version"] == "v1"
    assert block["arm_class"] == "krum_family"
    assert block["scenario_token"] == "S3"
    assert block["cut"] == 0.5
    assert block["serving_bundle_sha256"] == "ab" * 32
    assert block["n_rows"] == 2
    rows = block["rows"]
    assert rows[0]["malicious_gt"] is False
    assert rows[1]["malicious_gt"] is True
    assert rows[1]["score"] == 0.9


@pytest.mark.unit
def test_h2p_observe_block_none_when_not_observing():
    from run_phase4_flower import h2p_observe_block

    assert h2p_observe_block(_strategy([])) is None
    enforcing = _FakeDetector(observe_only=False, rows=[])
    assert h2p_observe_block(_strategy([enforcing])) is None
    assert h2p_observe_block(None) is None


@pytest.mark.unit
def test_h2p_observe_block_refuses_without_ground_truth_source():
    """A strategy with observe rows but no scenario adversary set cannot
    stamp malicious_gt — refusing beats silently labeling everyone honest
    (the builder pools 'honest' rows; mislabeling would move the cuts)."""
    from run_phase4_flower import h2p_observe_block

    det = _FakeDetector(rows=_ROWS)
    strategy = SimpleNamespace(_plugins=[det])  # no _adv_ids
    with pytest.raises(RuntimeError, match="_adv_ids|adversary"):
        h2p_observe_block(strategy)


# ---------------------------------------------------------------------------
# cache-reuse guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cuts_version_cache_guard_blocks_cross_version_reuse():
    """PR #68 P2: a cached ENFORCING v1 unit must NOT be silently reused for
    a --h2p-cuts-version v2 run (and vice versa)."""
    from run_phase4_flower import _h2p_cuts_version_cache_reusable

    v1_cached = {"provenance": {"h2p_cuts_version": "v1"}}
    v2_cached = {"provenance": {"h2p_cuts_version": "v2"}}
    legacy = {"provenance": {}}          # pre-erratum result: no key at all
    non_detector = {"provenance": {"h2p_cuts_version": None}}

    # cross-version: never reusable
    assert _h2p_cuts_version_cache_reusable(
        v1_cached, {"h2p-cuts-version": "v2"}) is False
    assert _h2p_cuts_version_cache_reusable(v2_cached, None) is False
    assert _h2p_cuts_version_cache_reusable(v2_cached, {}) is False
    # matching versions: reusable
    assert _h2p_cuts_version_cache_reusable(
        v2_cached, {"h2p-cuts-version": "v2"}) is True
    assert _h2p_cuts_version_cache_reusable(
        v1_cached, {"h2p-cuts-version": "v1"}) is True
    # active absent = the effective v1 default: a v1-cut unit matches it
    assert _h2p_cuts_version_cache_reusable(v1_cached, None) is True
    # legacy paths byte-unchanged: absent-vs-absent OK ...
    assert _h2p_cuts_version_cache_reusable(legacy, None) is True
    assert _h2p_cuts_version_cache_reusable(legacy, {}) is True
    # ... but a legacy cache never stands in for a DECLARED v2 run
    assert _h2p_cuts_version_cache_reusable(
        legacy, {"h2p-cuts-version": "v2"}) is False
    # non-detector arm (declared null): the knob cannot change its behavior
    assert _h2p_cuts_version_cache_reusable(non_detector, None) is True
    assert _h2p_cuts_version_cache_reusable(
        non_detector, {"h2p-cuts-version": "v2"}) is True


@pytest.mark.unit
def test_observe_cache_guard_blocks_cross_mode_reuse():
    from run_phase4_flower import _h2p_observe_cache_reusable

    observed = {"provenance": {"h2p_observe_only": True}}
    enforced = {"provenance": {"h2p_observe_only": False}}
    legacy = {"provenance": {}}

    assert _h2p_observe_cache_reusable(enforced, None) is True
    assert _h2p_observe_cache_reusable(legacy, {}) is True
    assert _h2p_observe_cache_reusable(
        observed, {"h2p-observe-only": True}) is True
    # cross-mode: never reuse an observe result for an enforcing run or
    # vice versa
    assert _h2p_observe_cache_reusable(observed, None) is False
    assert _h2p_observe_cache_reusable(
        enforced, {"h2p-observe-only": True}) is False
    assert _h2p_observe_cache_reusable(
        legacy, {"h2p-observe-only": True}) is False
