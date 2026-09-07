"""Unit tests for praxis_exp/mlflow_enrichment.py — pure helpers shared by
matrix_launch.py, docker/entrypoint.py, and ingest.py for MLflow S3-link
tags, dataset decoration, and experiment/run descriptions."""
from pathlib import Path

from praxis_exp.mlflow_enrichment import (
    build_experiment_description,
    build_parent_run_params,
    cloudwatch_log_url,
    current_amendment_filename,
    dataset_digest_from_metadata,
    dataset_source_uri,
    parent_run_s3_tags,
    unit_s3_tags,
)


class _Doc:
    def __init__(self, body, exp_id="EXP-005", slug="h2-dev-sweep",
                 methodology_version="v1.17"):
        self.exp_id = exp_id
        self.slug = slug
        self.methodology_version = methodology_version
        self.body = body


def _specs_repo(tmp_path, *names):
    specs = tmp_path / "docs" / "superpowers" / "specs"
    specs.mkdir(parents=True)
    for n in names:
        (specs / n).write_text("placeholder")
    return tmp_path


def test_current_amendment_filename_picks_latest_dated_amendment(tmp_path):
    repo = _specs_repo(
        tmp_path,
        "2026-05-27-praxis-experimental-design.md",
        "2026-06-05-praxis-experimental-design-v1.1.md",
        "2026-07-11-praxis-experimental-design-v1.6.md",
        "2026-06-09-praxis-experimental-design-v1.4.md",
    )
    assert current_amendment_filename(repo) == "2026-07-11-praxis-experimental-design-v1.6.md"


def test_current_amendment_filename_falls_back_when_no_specs_dir(tmp_path):
    assert current_amendment_filename(tmp_path) == "2026-05-27-praxis-experimental-design.md"


def test_current_amendment_filename_ignores_unrelated_files(tmp_path):
    repo = _specs_repo(
        tmp_path,
        "2026-05-27-praxis-experimental-design.md",
        "2026-06-06-praxis-fleet-harness-design.md",
    )
    assert current_amendment_filename(repo) == "2026-05-27-praxis-experimental-design.md"


def test_build_experiment_description_includes_title_paragraph_and_amendment(tmp_path):
    repo = _specs_repo(tmp_path, "2026-07-11-praxis-experimental-design-v1.6.md")
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    doc_path = exp_dir / "EXP-005-h2-dev-sweep.md"
    doc_path.write_text("front matter\n")
    doc = _Doc(body="# H2 Dev Sweep\n\nThis sweep tests Krum vs TrustScore under RMC.\n\nMore body.")
    desc = build_experiment_description(doc, repo, doc_path)
    assert "H2 Dev Sweep" in desc
    assert "This sweep tests Krum vs TrustScore under RMC." in desc
    assert "docs/experiments/EXP-005-h2-dev-sweep.md" in desc
    assert "v1.17" in desc and "docs/METHODOLOGY_LOG.md" in desc
    assert "2026-07-11-praxis-experimental-design-v1.6.md" in desc


def test_build_experiment_description_falls_back_title_when_no_heading(tmp_path):
    repo = _specs_repo(tmp_path, "2026-05-27-praxis-experimental-design.md")
    doc_path = tmp_path / "docs" / "experiments" / "EXP-005-h2-dev-sweep.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("x")
    doc = _Doc(body="Dev sweep.\n")
    desc = build_experiment_description(doc, repo, doc_path)
    assert "EXP-005" in desc and "h2-dev-sweep" in desc


def test_build_parent_run_params():
    doc = _Doc(body="x")
    doc.defenses = ["Krum", "TrustScore"]
    doc.scenarios = ["S0", "S4"]
    doc.seeds = [42, 137]
    doc.mode = "persistent_optimizer"
    doc.rounds = 50
    doc.max_per_client = 2_000_000
    doc.repeats = 1
    params = build_parent_run_params(doc, n_units=8)
    assert params == {
        "defenses": "Krum,TrustScore",
        "scenarios": "S0,S4",
        "seeds": "42,137",
        "mode": "persistent_optimizer",
        "rounds": "50",
        "max_per_client": "2000000",
        "repeats": "1",
        "n_units": "8",
    }


def _matrix_doc_yaml(repeats_line: str) -> str:
    """A minimal, valid matrix design doc; ``repeats_line`` is '' (axis absent)
    or a '  repeats: N\\n' line under the matrix block."""
    return (
        "---\n"
        "exp_id: EXP-005\n"
        "slug: h2-dev-sweep\n"
        "hypothesis: H2\n"
        "methodology_version: v1.17\n"
        "matrix:\n"
        "  defenses: [Krum, TrustScore]\n"
        "  scenarios: [S0, S4]\n"
        "  seeds: [42, 137]\n"
        "  mode: persistent_optimizer\n"
        "  max_per_client: 2000000\n"
        "  rounds: 50\n"
        f"{repeats_line}"
        "batch:\n"
        "  job_queue: q\n"
        "  job_definition: d\n"
        "---\n"
        "Dev sweep.\n"
    )


def test_build_parent_run_params_carries_repeats_axis(tmp_path):
    """the parent run must be able to reconstruct the full
    swept matrix from its params — including the replicate axis. The param is
    always present: a repeats-less design doc (parse_matrix default) records
    repeats=1, explicitly documenting the axis was inactive."""
    from praxis_exp.matrix_doc import parse_matrix

    p1 = tmp_path / "no_repeats.md"
    p1.write_text(_matrix_doc_yaml(""))
    assert build_parent_run_params(parse_matrix(p1), n_units=8)["repeats"] == "1"

    p2 = tmp_path / "repeats3.md"
    p2.write_text(_matrix_doc_yaml("  repeats: 3\n"))
    assert build_parent_run_params(parse_matrix(p2), n_units=24)["repeats"] == "3"


def test_parent_run_s3_tags():
    tags = parent_run_s3_tags("praxis-bucket", "EXP-005")
    assert tags["s3_manifest_uri"] == "s3://praxis-bucket/sweeps/EXP-005/manifest.json"
    assert tags["s3_console_url"] == (
        "https://us-east-1.console.aws.amazon.com/s3/buckets/praxis-bucket"
        "?prefix=sweeps/EXP-005/"
    )


def test_dataset_source_uri():
    assert dataset_source_uri("praxis-bucket") == "s3://praxis-bucket/data/edge_full_20/"
    assert dataset_source_uri("b", "edge_full") == "s3://b/data/edge_full/"


def test_dataset_digest_from_metadata_is_stable_and_optional():
    meta = {"_meta": {"total_rows": 20939617, "num_features": 45,
                      "partitioning": {"num_partitions": 20}}}
    digest = dataset_digest_from_metadata(meta)
    assert digest and digest == dataset_digest_from_metadata(meta)  # stable
    assert dataset_digest_from_metadata(None) is None
    assert dataset_digest_from_metadata({"_meta": {}}) is None


def test_cloudwatch_log_url_group_and_stream():
    group_url = cloudwatch_log_url(region="us-east-1")
    assert group_url == (
        "https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1"
        "#logsV2:log-groups/log-group/$252Faws$252Fbatch$252Fjob"
    )
    stream_url = cloudwatch_log_url(region="us-east-1", log_stream="jobdef/default/abc123")
    assert stream_url.endswith("/log-events/jobdef$252Fdefault$252Fabc123")


def test_unit_s3_tags():
    tags = unit_s3_tags("praxis-bucket", "EXP-005", "s0__krum__persistent_optimizer__seed42")
    assert tags["s3_result_uri"] == (
        "s3://praxis-bucket/sweeps/EXP-005/results/s0__krum__persistent_optimizer__seed42.json"
    )
    assert tags["s3_signal_uri"] == (
        "s3://praxis-bucket/sweeps/EXP-005/signals/s0__krum__persistent_optimizer__seed42.jsonl"
    )
    assert tags["s3_done_uri"] == (
        "s3://praxis-bucket/sweeps/EXP-005/done/s0__krum__persistent_optimizer__seed42.marker"
    )
    # No trailing slash: the console filter must string-match the actual
    # OBJECT key `.../results/{unit_id}.json` .
    assert tags["s3_console_url"] == (
        "https://us-east-1.console.aws.amazon.com/s3/buckets/praxis-bucket"
        "?prefix=sweeps/EXP-005/results/s0__krum__persistent_optimizer__seed42"
    )


def _console_prefix(url):
    from urllib.parse import parse_qs, urlsplit
    return parse_qs(urlsplit(url).query)["prefix"][0]


def test_unit_console_url_prefix_matches_a_real_key():
    """Drift-proof guard : the unit console
    URL's prefix param must be a true string-prefix of the result object's
    actual S3 key — both derived from the same storage functions, so the
    console listing can never be empty for a persisted unit."""
    from praxis_exp import storage
    unit_id = "s0__krum__persistent_optimizer__seed42"
    tags = unit_s3_tags("praxis-bucket", "EXP-005", unit_id)
    prefix = _console_prefix(tags["s3_console_url"])
    assert storage.result_key("EXP-005", unit_id).startswith(prefix)


def test_parent_console_url_prefix_matches_real_keys():
    """Same guard for the parent (sweep) console link: its prefix must be a
    true string-prefix of every artifact key persist_unit writes."""
    from praxis_exp import storage
    unit_id = "s0__krum__persistent_optimizer__seed42"
    tags = parent_run_s3_tags("praxis-bucket", "EXP-005")
    prefix = _console_prefix(tags["s3_console_url"])
    for key in (
        storage.manifest_key("EXP-005"),
        storage.result_key("EXP-005", unit_id),
        storage.signal_key("EXP-005", unit_id),
        storage.marker_key("EXP-005", unit_id),
    ):
        assert key.startswith(prefix)
