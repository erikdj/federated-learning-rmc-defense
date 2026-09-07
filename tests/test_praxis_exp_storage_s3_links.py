"""Unit tests for storage.py's S3 URI / console-URL helpers (req 5: MLflow
must carry functional links to the per-unit S3 artifacts, not just a bucket
name in a param)."""
from praxis_exp.storage import s3_console_url, s3_uri


def test_s3_uri_joins_bucket_and_key():
    assert s3_uri("praxis-bucket", "sweeps/EXP-005/manifest.json") == \
        "s3://praxis-bucket/sweeps/EXP-005/manifest.json"


def test_s3_uri_strips_leading_slash_on_key():
    assert s3_uri("praxis-bucket", "/sweeps/EXP-005/manifest.json") == \
        "s3://praxis-bucket/sweeps/EXP-005/manifest.json"


def test_s3_console_url_default_region():
    url = s3_console_url("praxis-bucket", "sweeps/EXP-005")
    assert url == "https://us-east-1.console.aws.amazon.com/s3/buckets/praxis-bucket?prefix=sweeps/EXP-005/"


def test_s3_console_url_custom_region():
    url = s3_console_url("praxis-bucket", "sweeps/EXP-005", region="us-west-2")
    assert url.startswith("https://us-west-2.console.aws.amazon.com/")


def test_s3_console_url_exact_prefix_keeps_no_trailing_slash():
    """ : a unit's console link must filter on
    a prefix that string-matches the OBJECT key `.../results/{unit_id}.json`.
    With exact=True the given prefix is used verbatim — no appended `/`
    (which would make the filter `.../{unit_id}/`, matching nothing)."""
    url = s3_console_url("praxis-bucket", "sweeps/EXP-005/results/unit_x", exact=True)
    assert url == (
        "https://us-east-1.console.aws.amazon.com/s3/buckets/praxis-bucket"
        "?prefix=sweeps/EXP-005/results/unit_x"
    )


def test_s3_console_url_default_normalizes_existing_trailing_slash():
    """Directory-style prefixes get exactly one trailing slash."""
    assert s3_console_url("b", "sweeps/EXP-005/") == \
        s3_console_url("b", "sweeps/EXP-005")
