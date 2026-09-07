"""Resolves the MLflow tracking URI and AWS-resource defaults for the CLI.

Site-specific values (artifact bucket, AWS profile) carry NO baked-in
defaults: the public release ships this module, and private account
identifiers must not ship with it. Resolution order per field:

    env var  >  explicit constructor value  >  praxis_exp/local_defaults
    (private module, excluded from the release)  >  ConfigError

AWS region resolution follows the SDK convention: ``AWS_REGION`` then
``AWS_DEFAULT_REGION`` then the constructor/dataclass default.
"""
import os
from dataclasses import dataclass
from typing import Optional


class ConfigError(RuntimeError):
    """Raised when required site configuration cannot be resolved."""


def _site_default(attr: str) -> Optional[str]:
    try:
        from praxis_exp import local_defaults  # private; not in the public release
    except ImportError:
        return None
    return getattr(local_defaults, attr, None)


@dataclass(frozen=True)
class Config:
    """Praxis-exp CLI configuration."""
    tracking_uri: str = "http://localhost:5001"
    # Separate from tracking_uri: the local client (operator's laptop) needs the
    # SSM-tunnel address above, but AWS Batch containers run inside the VPC and
    # can never reach that localhost. Resolution: env var > explicit value >
    # local_defaults.CONTAINER_MLFLOW_URI > tracking_uri. Unlike bucket/profile
    # below, this must NOT raise when unresolved — it silently falls back to
    # tracking_uri (matrix_launch.py guards against a localhost container URI).
    container_tracking_uri: str = ""
    artifact_bucket: str = ""
    aws_profile: str = ""
    aws_region: str = "us-east-1"

    def __post_init__(self) -> None:
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or self.aws_region
        )
        object.__setattr__(self, "aws_region", region)
        if env := os.environ.get("MLFLOW_TRACKING_URI"):
            object.__setattr__(self, "tracking_uri", env)
        container_tracking_uri = (
            os.environ.get("PRAXIS_CONTAINER_MLFLOW_URI")
            or self.container_tracking_uri
            or _site_default("CONTAINER_MLFLOW_URI")
            or self.tracking_uri
        )
        object.__setattr__(self, "container_tracking_uri", container_tracking_uri)
        # Role mode: a Lambda (or any instance-role execution) authenticates via
        # boto's default credential chain (its execution role), so a named
        # AWS_PROFILE is neither present nor needed. When PRAXIS_USE_INSTANCE_ROLE
        # is truthy — or the Lambda runtime marker AWS_LAMBDA_FUNCTION_NAME is set —
        # skip ONLY the aws_profile requirement (and leave it empty; the default
        # chain, not a profile, serves S3 reads and the mlflow.log_artifact upload).
        # artifact_bucket stays REQUIRED regardless (GWU-45 Lane B, design § 5.5).
        role_mode = bool(
            os.environ.get("PRAXIS_USE_INSTANCE_ROLE")
            or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        )
        required = [
            ("artifact_bucket", "PRAXIS_ARTIFACT_BUCKET", "ARTIFACT_BUCKET", "artifact bucket"),
        ]
        if not role_mode:
            required.append(("aws_profile", "AWS_PROFILE", "AWS_PROFILE", "AWS profile"))
        for attr, env_name, site_attr, what in required:
            value = os.environ.get(env_name) or getattr(self, attr) or _site_default(site_attr)
            if not value:
                raise ConfigError(
                    f"no {what} configured: set {env_name} "
                    "(or provide praxis_exp/local_defaults.py — private, not shipped)"
                )
            object.__setattr__(self, attr, value)
