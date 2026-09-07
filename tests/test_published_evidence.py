"""Published summaries must agree with their cells and frozen instruments."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[1] / "reproduction/verify_published_results.py"
    spec = importlib.util.spec_from_file_location("published_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_read(monkeypatch, verifier, name, document):
    original = verifier.read
    monkeypatch.setattr(verifier, "read", lambda filename: document if filename == name else original(filename))


def test_h1_published_development_frozen_read(verifier):
    verifier.verify_h1()


def test_h1_rejects_a_cell_with_a_different_operating_cut(verifier, monkeypatch):
    document = deepcopy(verifier.read("h1-heldout.json"))
    cell = next(iter(document["primary"]["per_cell"].values()))
    cell["C"]["threshold"] += 0.01
    replace_read(monkeypatch, verifier, "h1-heldout.json", document)
    with pytest.raises(AssertionError):
        verifier.verify_h1()


def test_h1_rejects_a_summary_that_disagrees_with_its_cells(verifier, monkeypatch):
    document = deepcopy(verifier.read("h1-heldout.json"))
    document["primary"]["per_scenario"]["S4"]["W"] += 0.01
    replace_read(monkeypatch, verifier, "h1-heldout.json", document)
    with pytest.raises(AssertionError):
        verifier.verify_h1()


def test_h2prime_rejects_a_sensitivity_with_a_changed_primary_reference(verifier, monkeypatch):
    document = deepcopy(verifier.read("h2prime-confirmatory.json"))
    document["secondaries"]["window_aware_loao_sensitivity"]["p1_quantity"]["primary"]["mean_recall"] += 0.01
    replace_read(monkeypatch, verifier, "h2prime-confirmatory.json", document)
    with pytest.raises(AssertionError):
        verifier.verify_h2prime()


def test_provenance_rejects_a_stale_calibration_hash(verifier, monkeypatch):
    document = deepcopy(verifier.read("provenance.json"))
    document["calibration_artifact"]["public_sha256"] = "0" * 64
    replace_read(monkeypatch, verifier, "provenance.json", document)
    with pytest.raises(AssertionError):
        verifier.verify_provenance()
