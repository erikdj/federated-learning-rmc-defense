"""AWS Batch array-job submission abstraction (fake for tests, boto3 for prod)."""
from __future__ import annotations

from typing import Any, Protocol


class BatchSubmitter(Protocol):
    def submit_array(
        self, *, job_name: str, job_queue: str, job_definition: str,
        size: int, environment: dict[str, str],
        tags: dict[str, str] | None = None,
    ) -> str: ...

    def job_definition_image(self, job_definition: str) -> str | None:
        """The container image reference the job definition actually runs, or
        ``None`` when this submitter does not implement resolution (a test fake
        opting out of image verification).

        GWU-48: ``--image-digest`` is provenance-only — it does not select the
        image AWS Batch runs (the job definition's ``containerProperties.image``
        does). launch-matrix compares the two before submitting so a launch on a
        stale job def fails fast instead of silently running the wrong image. A
        production submitter MUST return a concrete reference or raise; ``None``
        is reserved for the opt-out fake so the guard is skipped, never faked.
        """
        ...

    def array_terminal(self, array_job_id: str) -> bool:
        """True iff the given array job is in a terminal state (SUCCEEDED /
        FAILED) with no attempt still able to run.

        GWU-41 invariant 5: a provenance-changing refill must not launch while
        the prior array is still RUNNABLE/RUNNING — an old child could win the
        done-marker race for a still-missing cell under the OLD image while the
        refill advertises new provenance.
        """
        ...


class FakeBatchSubmitter:
    """Records submissions instead of calling AWS."""

    def __init__(self, job_def_image: str | None = None, array_terminal: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        # None -> opt out of GWU-48 image verification (default: the vast
        # majority of launch tests do not exercise the guard). Set to a
        # concrete "…@sha256:…" (or tag-pinned) reference to drive the guard.
        self.job_def_image = job_def_image
        # GWU-41 invariant 5: default terminal (the storm-recovery case).
        self._array_terminal = array_terminal
        self.array_terminal_queries: list[str] = []

    def submit_array(
        self, *, job_name: str, job_queue: str, job_definition: str,
        size: int, environment: dict[str, str],
        tags: dict[str, str] | None = None,
    ) -> str:
        self.calls.append({
            "job_name": job_name, "job_queue": job_queue, "job_definition": job_definition,
            "size": size, "environment": dict(environment),
            "tags": dict(tags) if tags else None,
        })
        return "fake-array-job-id"

    def job_definition_image(self, job_definition: str) -> str | None:
        return self.job_def_image

    def array_terminal(self, array_job_id: str) -> bool:
        # Records the query so tests can assert the check ran; default True
        # (the common case: refilling a storm-exhausted, terminal array).
        self.array_terminal_queries.append(array_job_id)
        return self._array_terminal


class Boto3BatchSubmitter:
    """boto3-backed submitter. `client` is a boto3 'batch' client (injected).

    Note: AWS Batch rejects array size < 2. The caller (launch_matrix) validates
    size before calling submit_array; this class is a thin pass-through.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def submit_array(
        self, *, job_name: str, job_queue: str, job_definition: str,
        size: int, environment: dict[str, str],
        tags: dict[str, str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "jobName": job_name,
            "jobQueue": job_queue,
            "jobDefinition": job_definition,
            "arrayProperties": {"size": size},
            "containerOverrides": {
                "environment": [{"name": k, "value": str(v)} for k, v in environment.items()],
            },
        }
        if tags:
            # Job-resource tags (visible in the Batch console + cost explorer);
            # propagateTags carries them onto the ECS task so devops can trace
            # instance -> task -> job -> EXP in one hop.
            kwargs["tags"] = {k: str(v) for k, v in tags.items()}
            kwargs["propagateTags"] = True
        resp = self._client.submit_job(**kwargs)
        return resp["jobId"]

    def job_definition_image(self, job_definition: str) -> str | None:
        """Resolve the job definition's ``containerProperties.image`` via
        ``batch:describe-job-definitions`` (GWU-48).

        Never returns ``None`` in production: an unresolvable job definition or
        a definition without a container image is unaccountable state for a
        launch about to run on it, so both raise rather than silently disabling
        the digest guard (``None`` is the fake's opt-out only).
        """
        resp = self._client.describe_job_definitions(jobDefinitions=[job_definition])
        defs = resp.get("jobDefinitions", [])
        if not defs:
            raise RuntimeError(
                f"batch:describe-job-definitions returned no job definition for "
                f"{job_definition!r} — cannot verify the container image digest"
            )
        image = defs[0].get("containerProperties", {}).get("image")
        if not image:
            raise RuntimeError(
                f"job definition {job_definition!r} has no containerProperties.image — "
                "cannot verify the container image digest"
            )
        return image

    # Terminal Batch array-job states (no attempt can still run).
    _TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED"})

    def array_terminal(self, array_job_id: str) -> bool:
        resp = self._client.describe_jobs(jobs=[array_job_id])
        jobs = resp.get("jobs", [])
        if not jobs:
            raise RuntimeError(
                f"batch:describe-jobs returned no job for array id {array_job_id!r} — "
                "cannot verify the prior array is terminal before a provenance-changing refill"
            )
        return jobs[0].get("status") in self._TERMINAL_STATES
