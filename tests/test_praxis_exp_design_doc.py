"""Design-doc parser tests."""
from pathlib import Path
import pytest


VALID_DOC = """---
exp_id: EXP-001
slug: my-experiment
hypothesis: Krum should degrade under RMC
methodology_version: v1.2
params:
  defense: Krum
  scenario: rmc/scenarios/rmc_intensity_9_continuous_v2.json
  seed: 42
  mode: flower_reset
  max_per_client: 2000000
predictions:
  final_accuracy_min: 0.4
  final_accuracy_max: 0.7
---

# EXP-001 — my-experiment

Body of design doc here.
"""


def test_parses_valid_doc(tmp_path):
    from praxis_exp.design_doc import parse
    p = tmp_path / "EXP-001-my-experiment.md"
    p.write_text(VALID_DOC)
    doc = parse(p)
    assert doc.exp_id == "EXP-001"
    assert doc.slug == "my-experiment"
    assert doc.hypothesis.startswith("Krum should")
    assert doc.methodology_version == "v1.2"
    assert doc.params["defense"] == "Krum"
    assert doc.predictions["final_accuracy_min"] == 0.4


def test_rejects_missing_frontmatter(tmp_path):
    from praxis_exp.design_doc import parse, DesignDocError
    p = tmp_path / "EXP-001-bad.md"
    p.write_text("# no front matter here")
    with pytest.raises(DesignDocError, match="front-matter"):
        parse(p)


def test_rejects_doc_with_tbd_marker(tmp_path):
    """Body containing TBD/TODO/<FILL> must fail validation."""
    from praxis_exp.design_doc import parse, DesignDocError
    p = tmp_path / "EXP-001-tbd.md"
    p.write_text(VALID_DOC.replace("Body of design doc here.", "TBD: write this section"))
    with pytest.raises(DesignDocError, match="placeholder"):
        parse(p)


def test_rejects_missing_required_field(tmp_path):
    from praxis_exp.design_doc import parse, DesignDocError
    p = tmp_path / "EXP-001-bad.md"
    p.write_text(VALID_DOC.replace("hypothesis: Krum should degrade under RMC\n", ""))
    with pytest.raises(DesignDocError, match="hypothesis"):
        parse(p)
