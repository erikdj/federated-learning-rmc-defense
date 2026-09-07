"""The container env must satisfy docker/entrypoint.py's runtime imports.

environment-aws.yml is the ONLY dependency source in docker/Dockerfile
(pyproject deps are never installed in the image), and entrypoint.py imports
boto3/mlflow lazily inside main() so unit tests import cleanly — which also
means no test ever exercised those imports. This contract test keeps the
locked env honest without building the image.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def _pinned_pip_packages() -> set[str]:
    text = (REPO / "environment-aws.yml").read_text()
    return {m.group(1).lower() for m in re.finditer(r"^\s*-\s*([A-Za-z0-9_.-]+)==", text, re.M)}


def _pinned_pip_versions() -> dict[str, str]:
    text = (REPO / "environment-aws.yml").read_text()
    return {
        m.group(1).lower(): m.group(2)
        for m in re.finditer(r"^\s*-\s*([A-Za-z0-9_.-]+)==(\S+)", text, re.M)
    }


# The FL-numerics pins whose byte-identical value gates every cross-image result
# comparison. pandas is included — it is the data-load engine (flowerfl/task.py:625
# parquet read, :637-642 seeded groupby/sample shape the training data). pyarrow is
# deliberately EXCLUDED: it is the one allowed mover (24.0.0 -> <=23.x under full
# mlflow's `pyarrow<24`), guarded instead by the data-load equivalence gate
# (GWU-46 Task 5a), not a fixed version bound.
FL_CRITICAL_PINS: dict[str, str] = {
    "torch": "2.11.0+cpu",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "ray": "2.51.1",
    "flwr": "1.29.0",
    "xgboost": "3.2.0",
    "numba": "0.65.1",
    "llvmlite": "0.47.0",
    "pandas": "2.3.3",
}


def test_entrypoint_lazy_imports_still_present():
    # Guard the premise: if the lazy imports move/vanish, revisit this contract.
    src = (REPO / "docker" / "entrypoint.py").read_text()
    lazy = set(re.findall(r"^\s+import (\w+)", src, re.M))
    assert {"boto3", "mlflow"} <= lazy


def test_entrypoint_runtime_deps_are_pinned_in_aws_env():
    pins = _pinned_pip_packages()
    # The container needs full `mlflow` for the `mlflow.pytorch` flavor
    # (docker/entrypoint.py::log_native_model); mlflow-skinny lacks it. pyarrow
    # is NOT a training-numerics pin — it is pandas' parquet read engine
    # (flowerfl/task.py:571,625), so its full-mlflow downgrade (24.0.0 -> <=23.x)
    # is guarded by the data-load equivalence gate (GWU-46 Task 5a), not a version
    # bound. The has_mlflow OR stays correct — bare `mlflow` is now pinned too.
    has_mlflow = "mlflow" in pins or "mlflow-skinny" in pins
    assert "boto3" in pins and has_mlflow, (
        "docker/entrypoint.py runtime deps missing from environment-aws.yml "
        "— the gold image would die with ModuleNotFoundError at Batch job start"
    )


def test_fl_critical_pins_locked():
    # FL-pin-invariance gate (GWU-46): the FL-numerics pins + pandas must stay
    # byte-identical across images, or cross-image result comparisons are invalid.
    versions = _pinned_pip_versions()
    for pkg, expected in FL_CRITICAL_PINS.items():
        assert versions.get(pkg) == expected, (
            f"FL-critical pin moved: {pkg} {versions.get(pkg)} != {expected} "
            "— breaks cross-image result comparability (GWU-46 FL-pin-invariance gate)"
        )


def test_mlflow_pytorch_flavor_available():
    # The container's docker/entrypoint.py::log_native_model calls
    # mlflow.pytorch.log_model — a flavor module shipped ONLY by full `mlflow`,
    # not mlflow-skinny. Pinning bare `mlflow` (which pulls mlflow-skinny +
    # mlflow-tracing) is the native-model regression this issue fixes.
    pins = _pinned_pip_packages()
    assert "mlflow" in pins, (
        "full mlflow (with the pytorch flavor) must be pinned, not only "
        "mlflow-skinny — else log_native_model dies with ModuleNotFoundError "
        "on mlflow.pytorch and no native model is logged in-container"
    )
