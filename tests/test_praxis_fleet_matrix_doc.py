import textwrap
import pytest
from praxis_exp.matrix_doc import parse_matrix, MatrixDocError

VALID = textwrap.dedent('''\
    ---
    exp_id: EXP-TEST
    slug: h2-dev-sweep
    hypothesis: H2
    methodology_version: v1.9
    matrix:
      defenses: [Krum, TrustScore, Krum+TGE, FedAvg+TGE]
      scenarios: [S0, S1, S2, S3, S4]
      seeds: [42, 137, 256, 314, 500]
      mode: persistent_optimizer
      max_per_client: 2000000
      rounds: 50
    batch:
      job_queue: praxis-spot-queue
      job_definition: praxis-flowerfl-unit
    ---
    Dev-phase sweep for H2 frozen-threshold calibration.
    ''')


def test_parse_valid_matrix(tmp_path):
    p = tmp_path / "doc.md"; p.write_text(VALID)
    doc = parse_matrix(p)
    assert doc.exp_id == "EXP-TEST"
    assert doc.defenses == ["Krum", "TrustScore", "Krum+TGE", "FedAvg+TGE"]
    assert doc.seeds == [42, 137, 256, 314, 500]
    assert doc.max_per_client == 2_000_000
    assert doc.job_queue == "praxis-spot-queue"


def test_missing_matrix_field_raises(tmp_path):
    p = tmp_path / "doc.md"; p.write_text(VALID.replace("  seeds: [42, 137, 256, 314, 500]\n", ""))
    with pytest.raises(MatrixDocError, match="seeds"):
        parse_matrix(p)


def test_placeholder_body_rejected(tmp_path):
    p = tmp_path / "doc.md"; p.write_text(VALID.replace("Dev-phase sweep", "TODO write this"))
    with pytest.raises(MatrixDocError, match="placeholder"):
        parse_matrix(p)


def test_null_matrix_block_raises(tmp_path):
    doc = "---\nexp_id: E\nslug: s\nhypothesis: H2\nmethodology_version: v1.9\nmatrix:\nbatch:\n  job_queue: q\n  job_definition: d\n---\nbody\n"
    p = tmp_path / "doc.md"; p.write_text(doc)
    with pytest.raises(MatrixDocError, match="matrix"):
        parse_matrix(p)


def test_placeholder_in_frontmatter_rejected(tmp_path):
    p = tmp_path / "doc.md"; p.write_text(VALID.replace("job_queue: praxis-spot-queue", "job_queue: TBD"))
    with pytest.raises(MatrixDocError, match="placeholder"):
        parse_matrix(p)


# --- : optional matrix.repeats -------------------------------
# repeats is OPTIONAL (defaults to 1) — it must NOT join the required-field
# validator, or every existing design doc without it would fail to parse.

def test_repeats_defaults_to_one_when_absent(tmp_path):
    p = tmp_path / "doc.md"; p.write_text(VALID)
    doc = parse_matrix(p)
    assert doc.repeats == 1


def test_repeats_parsed_when_present(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(VALID.replace("  rounds: 50\n", "  rounds: 50\n  repeats: 5\n"))
    doc = parse_matrix(p)
    assert doc.repeats == 5


def test_invalid_repeats_rejected(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(VALID.replace("  rounds: 50\n", "  rounds: 50\n  repeats: 0\n"))
    with pytest.raises(MatrixDocError, match="repeats"):
        parse_matrix(p)


def test_float_repeats_rejected(tmp_path):
    """A sealed design doc must fail fast, not silently truncate: int(2.7)==2
    would quietly alter the replicate count."""
    p = tmp_path / "doc.md"
    p.write_text(VALID.replace("  rounds: 50\n", "  rounds: 50\n  repeats: 2.7\n"))
    with pytest.raises(MatrixDocError, match="repeats"):
        parse_matrix(p)


def test_bool_repeats_rejected(tmp_path):
    """YAML `true` is a Python bool (an int subclass); int(True)==1 would be
    silently accepted. Reject non-int types explicitly."""
    p = tmp_path / "doc.md"
    p.write_text(VALID.replace("  rounds: 50\n", "  rounds: 50\n  repeats: true\n"))
    with pytest.raises(MatrixDocError, match="repeats"):
        parse_matrix(p)
