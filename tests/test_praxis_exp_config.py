"""Config module tests.

The public release must contain no private AWS defaults, so Config resolves
bucket/profile as: env var > explicit value > praxis_exp/local_defaults
(private, unshipped) > ConfigError. These tests run in both the working repo
(local_defaults present) and the public export (absent).
"""
import sys

import pytest


def _clear_env(monkeypatch):
    for var in (
        "MLFLOW_TRACKING_URI",
        "PRAXIS_ARTIFACT_BUCKET",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "PRAXIS_CONTAINER_MLFLOW_URI",
        "PRAXIS_USE_INSTANCE_ROLE",
        "AWS_LAMBDA_FUNCTION_NAME",
    ):
        monkeypatch.delenv(var, raising=False)


def _configured_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_PROFILE", "test-profile")


def test_default_tracking_uri_is_localhost_5001(monkeypatch):
    """Default tracking URI is the local SSM forward port."""
    from praxis_exp.config import Config
    _configured_env(monkeypatch)
    assert Config().tracking_uri == "http://localhost:5001"


def test_env_overrides_tracking_uri(monkeypatch):
    from praxis_exp.config import Config
    _configured_env(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://test-server:9999")
    assert Config().tracking_uri == "http://test-server:9999"


def test_env_configures_bucket_and_profile(monkeypatch):
    from praxis_exp.config import Config
    _configured_env(monkeypatch)
    cfg = Config()
    assert cfg.artifact_bucket == "test-bucket"
    assert cfg.aws_profile == "test-profile"


def test_explicit_values_used_when_env_unset(monkeypatch):
    from praxis_exp.config import Config
    _clear_env(monkeypatch)
    cfg = Config(artifact_bucket="explicit-bucket", aws_profile="explicit-profile")
    assert cfg.artifact_bucket == "explicit-bucket"
    assert cfg.aws_profile == "explicit-profile"


def test_site_defaults_fill_when_env_unset(monkeypatch):
    local = pytest.importorskip("praxis_exp.local_defaults")  # absent in the public release
    from praxis_exp.config import Config
    _clear_env(monkeypatch)
    cfg = Config()
    assert cfg.artifact_bucket == local.ARTIFACT_BUCKET
    assert cfg.aws_profile == local.AWS_PROFILE


def test_raises_without_env_or_site_defaults(monkeypatch):
    import praxis_exp
    from praxis_exp.config import Config, ConfigError
    _clear_env(monkeypatch)
    # Hide the private module both ways `from praxis_exp import local_defaults`
    # can resolve it: the package attribute (set by any earlier import) and the
    # sys.modules entry (None makes a fresh import raise ImportError).
    monkeypatch.delattr(praxis_exp, "local_defaults", raising=False)
    monkeypatch.setitem(sys.modules, "praxis_exp.local_defaults", None)
    with pytest.raises(ConfigError):
        Config()


def test_config_role_mode_allows_absent_profile(monkeypatch):
    """GWU-45 Lane B: a Lambda authenticates via its execution role (boto default
    chain), so PRAXIS_USE_INSTANCE_ROLE=1 makes aws_profile optional — Config()
    constructs with NO AWS_PROFILE and leaves aws_profile empty. The artifact
    bucket stays REQUIRED even in role mode (design § 5.5)."""
    import sys
    import praxis_exp
    from praxis_exp.config import Config, ConfigError
    _clear_env(monkeypatch)
    monkeypatch.setenv("PRAXIS_USE_INSTANCE_ROLE", "1")
    monkeypatch.setenv("PRAXIS_ARTIFACT_BUCKET", "role-bucket")
    # No AWS_PROFILE and no need for one: role mode must not raise.
    cfg = Config()
    assert cfg.artifact_bucket == "role-bucket"
    assert cfg.aws_profile == ""  # execution role, not a named profile

    # Bucket is still required in role mode: hide every source and it must raise.
    monkeypatch.delenv("PRAXIS_ARTIFACT_BUCKET", raising=False)
    monkeypatch.delattr(praxis_exp, "local_defaults", raising=False)
    monkeypatch.setitem(sys.modules, "praxis_exp.local_defaults", None)
    with pytest.raises(ConfigError):
        Config()


# --- container_tracking_uri resolution -------------------------------------
# Batch containers run inside the VPC and can never reach the operator's
# localhost SSM tunnel; container_tracking_uri must resolve independently of
# tracking_uri, but unlike bucket/profile it must NOT raise when unresolved —
# it silently falls back to tracking_uri (the localhost guard lives in
# matrix_launch.py, not here).


def test_container_tracking_uri_env_overrides(monkeypatch):
    from praxis_exp.config import Config
    _configured_env(monkeypatch)
    monkeypatch.setenv("PRAXIS_CONTAINER_MLFLOW_URI", "http://10.99.9.9:5000")
    assert Config().container_tracking_uri == "http://10.99.9.9:5000"


def test_container_tracking_uri_falls_back_to_site_default(monkeypatch):
    local = pytest.importorskip("praxis_exp.local_defaults")  # absent in the public release
    from praxis_exp.config import Config
    _configured_env(monkeypatch)
    monkeypatch.delenv("PRAXIS_CONTAINER_MLFLOW_URI", raising=False)
    assert Config().container_tracking_uri == local.CONTAINER_MLFLOW_URI


def test_container_tracking_uri_falls_back_to_tracking_uri_without_site_default(monkeypatch):
    import praxis_exp
    from praxis_exp.config import Config
    _configured_env(monkeypatch)
    monkeypatch.delenv("PRAXIS_CONTAINER_MLFLOW_URI", raising=False)
    # Hide the private module the same way test_raises_without_env_or_site_defaults does.
    monkeypatch.delattr(praxis_exp, "local_defaults", raising=False)
    monkeypatch.setitem(sys.modules, "praxis_exp.local_defaults", None)
    cfg = Config()
    assert cfg.container_tracking_uri == cfg.tracking_uri


def test_aws_region_environment_overrides_default(monkeypatch):
    from praxis_exp.config import Config

    _configured_env(monkeypatch)
    monkeypatch.setenv('AWS_REGION', 'us-west-2')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'eu-west-1')
    assert Config().aws_region == 'us-west-2'


def test_aws_default_region_is_used_without_aws_region(monkeypatch):
    from praxis_exp.config import Config

    _configured_env(monkeypatch)
    monkeypatch.delenv('AWS_REGION', raising=False)
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'eu-west-1')
    assert Config().aws_region == 'eu-west-1'
