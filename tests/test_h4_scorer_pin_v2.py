"""Erratum-B build item 5: the H4 scorer's serving-bundle pin swap.

`docs/reproduction/experiments.md` (RULED,
methodology v1.53) retires the v1 corpus-quantile cuts for the sealed fleet:
the scorer's custody gate moves from the v1 bundle sha to the (not yet built)
bundle v2 sha. Until the real v2 sha is pinned post-calibration, the active
pin is an explicit sentinel and the scorer must REFUSE — loudly, naming
erratum B — rather than score a fleet against either the stale v1 pin or a
placeholder.

The v1 value is retained as `SERVING_BUNDLE_SHA256_V1` for interpreting
EXP-061 artifacts; it is NOT accepted for the sealed fleet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import h4_scoring_lib as lib  # noqa: E402

V1_SHA = "56eca8e31d1352a73e70962d2d3375e412def6f983bae6be29a7897fef16db8a"
FAKE_V2_SHA = "ab" * 32


@pytest.mark.unit
def test_v1_pin_retained_under_its_own_name():
    """The v1 bundle sha survives — for EXP-061 artifact interpretation only."""
    assert lib.SERVING_BUNDLE_SHA256_V1 == V1_SHA


@pytest.mark.unit
def test_active_pin_is_the_committed_bundle_v2_sha():
    """Post-swap : the active pin IS the sha256 of the committed
    data/h4_serving/manifest_v2.json bytes — recomputed here, never trusted."""
    import hashlib
    manifest = (Path(__file__).resolve().parent.parent
                / "data" / "h4_serving" / "manifest_v2.json")
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert lib.SERVING_BUNDLE_SHA256 == actual
    assert lib.SERVING_BUNDLE_SHA256 != lib.SERVING_BUNDLE_SHA256_V1
    assert lib.required_serving_bundle_sha256() == actual


@pytest.mark.unit
def test_sentinel_refuses_loudly_naming_erratum_b(monkeypatch):
    """The sentinel refusal path survives the swap (regression-guarded via
    monkeypatch now that the real pin is live)."""
    monkeypatch.setattr(lib, "SERVING_BUNDLE_SHA256", "TBD_BUNDLE_V2")
    with pytest.raises(lib.ScoringError) as exc:
        lib.required_serving_bundle_sha256()
    msg = str(exc.value)
    assert "erratum B" in msg
    assert "TBD_BUNDLE_V2" in msg


@pytest.mark.unit
def test_pinned_value_is_returned_once_swapped(monkeypatch):
    monkeypatch.setattr(lib, "SERVING_BUNDLE_SHA256", FAKE_V2_SHA)
    assert lib.required_serving_bundle_sha256() == FAKE_V2_SHA


@pytest.mark.unit
def test_a_malformed_pin_is_refused_not_served(monkeypatch):
    """Anything that is not a 64-hex sha256 refuses — a pin typo must never
    become the custody bar the whole sealed fleet is checked against."""
    monkeypatch.setattr(lib, "SERVING_BUNDLE_SHA256", "not-a-sha")
    with pytest.raises(lib.ScoringError):
        lib.required_serving_bundle_sha256()


@pytest.mark.unit
def test_scorer_custody_refuses_v1_sha_once_v2_is_pinned(monkeypatch, tmp_path):
    """A unit carrying the V1 bundle sha is a different instrument for the
    sealed fleet: with the pin swapped to (a fake) v2, custody equality
    refuses the v1 value."""
    from tests import h4_factory as fac
    from scripts import analyze_h4_composition as cli

    monkeypatch.setattr(lib, "SERVING_BUNDLE_SHA256", FAKE_V2_SHA)
    fac.install_split_manifest(tmp_path, monkeypatch)
    unit = fac.make_unit("h2p_fp_krum", "S1", 42)
    unit["provenance"]["serving_bundle_sha256"] = lib.SERVING_BUNDLE_SHA256_V1
    with pytest.raises(lib.ScoringError) as exc:
        cli._validate_unit(
            tmp_path / "u.json", unit, lib.split_manifest_sha256()
        )
    assert "serving_bundle_sha256" in str(exc.value)


@pytest.mark.unit
def test_scorer_custody_refuses_while_pin_is_the_sentinel(monkeypatch, tmp_path):
    """With the sentinel active, a detector-arm unit cannot be custody-checked
    at all: the refusal happens BEFORE any equality, naming erratum B.

    Post-swap: the sentinel state is restored via
    monkeypatch, and the unit carries the REAL v2 pin — so the only way this
    can refuse is through the sentinel accessor, never an ordinary
    hash-mismatch that happens to share message text."""
    from tests import h4_factory as fac
    from scripts import analyze_h4_composition as cli

    monkeypatch.setattr(lib, "SERVING_BUNDLE_SHA256", "TBD_BUNDLE_V2")
    fac.install_split_manifest(tmp_path, monkeypatch)
    unit = fac.make_unit("h2p_fp_krum", "S1", 42)
    unit["provenance"]["serving_bundle_sha256"] = (
        "3bfeefb45700dff2e02a14acb0de4acfadcc717a7b82d6c246d44a8f44fa062c"
    )
    with pytest.raises(lib.ScoringError) as exc:
        cli._validate_unit(
            tmp_path / "u.json", unit, lib.split_manifest_sha256()
        )
    msg = str(exc.value)
    assert "erratum B" in msg
    assert "TBD_BUNDLE_V2" in msg  # the sentinel refusal, not a mismatch


def test_scorer_custody_refuses_stale_v1_pin_against_live_v2(tmp_path, monkeypatch):
    """With the REAL v2 pin live, a v1-bundled unit refuses as an ordinary
    custody mismatch (the EXP-061-era bundle is not the fleet instrument)."""
    from tests import h4_factory as fac
    from scripts import analyze_h4_composition as cli

    fac.install_split_manifest(tmp_path, monkeypatch)
    unit = fac.make_unit("h2p_fp_krum", "S1", 42)
    unit["provenance"]["serving_bundle_sha256"] = lib.SERVING_BUNDLE_SHA256_V1
    with pytest.raises(lib.ScoringError) as exc:
        cli._validate_unit(
            tmp_path / "u.json", unit, lib.split_manifest_sha256()
        )
    assert "TBD_BUNDLE_V2" not in str(exc.value)  # mismatch path, not sentinel
