"""Object-store abstraction for sweep artifacts (in-memory fake + boto3 S3)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from botocore.exceptions import ClientError


class ObjectExistsError(RuntimeError):
    """Raised when a conditional (if_none_match) put hits an existing key."""


class ObjectNotFoundError(RuntimeError):
    """Raised when get_bytes targets a key that does not exist."""


def sweep_prefix(exp_id: str) -> str:
    return f"sweeps/{exp_id}"


def result_key(exp_id: str, unit_id: str) -> str:
    return f"{sweep_prefix(exp_id)}/results/{unit_id}.json"


def signal_key(exp_id: str, unit_id: str) -> str:
    return f"{sweep_prefix(exp_id)}/signals/{unit_id}.jsonl"


def marker_key(exp_id: str, unit_id: str) -> str:
    return f"{sweep_prefix(exp_id)}/done/{unit_id}.marker"


def manifest_key(exp_id: str) -> str:
    return f"{sweep_prefix(exp_id)}/manifest.json"


def s3_uri(bucket: str, key: str) -> str:
    """A functional s3:// URI for a bucket/key pair (copy-pasteable into the
    AWS CLI or SDKs). Single source of truth so callers never hand-roll
    ``f"s3://{bucket}/{key}"`` themselves."""
    return f"s3://{bucket}/{key.lstrip('/')}"


def s3_console_url(bucket: str, prefix: str, *, region: str = "us-east-1", exact: bool = False) -> str:
    """A clickable AWS S3 console deep-link scoped to ``prefix`` within ``bucket``.

    Used to make MLflow a "functional directory" of experiments: every run
    that touches S3 gets both the raw ``s3://`` URI (for tooling) and this
    HTTP link (for a human clicking through from the MLflow UI).

    The console's ``prefix`` param is a plain string-prefix filter over keys.
    Default (``exact=False``): directory semantics — exactly one trailing
    slash is appended, for prefixes that name a "folder" of objects. Pass
    ``exact=True`` when the prefix must string-match an OBJECT key stem
    (e.g. ``.../results/{unit_id}`` matching ``{unit_id}.json``) — appending
    ``/`` there would make the filter match nothing (PR #13 P2,
    comment 3566953651).
    """
    p = prefix.lstrip("/")
    if not exact:
        p = p.rstrip("/") + "/"
    return f"https://{region}.console.aws.amazon.com/s3/buckets/{bucket}?prefix={p}"


class ObjectStore(Protocol):
    def put_bytes(self, key: str, data: bytes, *, if_none_match: bool = False) -> None: ...
    def put_file(self, key: str, path: Path) -> None: ...
    def head(self, key: str) -> bool: ...

    def size(self, key: str) -> int | None:
        """Return the object's byte length, or None if the key is absent."""
        ...

    def get_bytes(self, key: str) -> bytes:
        """Return the object bytes; raise ObjectNotFoundError if the key is absent."""
        ...

    def delete(self, key: str) -> None:
        """Delete the object; deleting an absent key is a no-op (idempotent).

        Added for launch rollback (PR #13 P2, comment 3567167519): a failed
        launch deletes the manifest it wrote so the retry is not refused by
        the one-launch-per-EXP guard.
        """
        ...

    def find_keys(self, prefix: str, *, limit: int = 25) -> list[str]:
        """Up to ``limit`` keys under ``prefix``, lexicographic order.

        Added for the namespace-emptiness pre-flight (PR #13 P2, comment
        3567187757): launch freshness must be prefix-level — a unit-id-keyed
        probe has blind spots the moment the matrix changes.
        """
        ...


class InMemoryObjectStore:
    """Test fake with the same semantics the S3 store must honour."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes, *, if_none_match: bool = False) -> None:
        if if_none_match and key in self._data:
            raise ObjectExistsError(key)
        self._data[key] = bytes(data)

    def put_file(self, key: str, path: Path) -> None:
        self._data[key] = Path(path).read_bytes()

    def head(self, key: str) -> bool:
        return key in self._data

    def size(self, key: str) -> int | None:
        data = self._data.get(key)
        return None if data is None else len(data)

    def get_bytes(self, key: str) -> bytes:
        if key not in self._data:
            raise ObjectNotFoundError(key)
        return self._data[key]

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def find_keys(self, prefix: str, *, limit: int = 25) -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))[:limit]


# ---------------------------------------------------------------------------
# boto3-backed store
# ---------------------------------------------------------------------------

_NOT_FOUND = {"NoSuchKey", "404"}      # NoSuchKey: GetObject; 404: HeadObject. (dropped non-S3 "NotFound")
_PRECONDITION = {"PreconditionFailed"}  # S3 PutObject IfNoneMatch violation. (dropped "412": HTTP status, never the Error.Code)


class S3ObjectStore:
    """boto3-backed ObjectStore. `client` is a boto3 s3 client (injected)."""

    def __init__(self, bucket: str, client: Any) -> None:
        self._bucket = bucket
        self._client = client

    def put_bytes(self, key: str, data: bytes, *, if_none_match: bool = False) -> None:
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Key": key, "Body": data}
        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        try:
            self._client.put_object(**kwargs)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _PRECONDITION:
                raise ObjectExistsError(key) from exc
            raise

    def put_file(self, key: str, path: Path) -> None:
        self._client.upload_file(str(path), self._bucket, key)

    def head(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _NOT_FOUND:
                return False
            raise

    def size(self, key: str) -> int | None:
        try:
            return self._client.head_object(Bucket=self._bucket, Key=key)["ContentLength"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _NOT_FOUND:
                return None
            raise

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _NOT_FOUND:
                raise ObjectNotFoundError(key) from exc
            raise

    def delete(self, key: str) -> None:
        # S3 DeleteObject is idempotent (204 even when the key is absent).
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def find_keys(self, prefix: str, *, limit: int = 25) -> list[str]:
        resp = self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=prefix, MaxKeys=limit
        )
        return [obj["Key"] for obj in resp.get("Contents", [])]
