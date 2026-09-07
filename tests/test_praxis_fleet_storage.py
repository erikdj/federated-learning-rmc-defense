import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
from praxis_exp import storage
from praxis_exp.storage import InMemoryObjectStore, ObjectExistsError, ObjectNotFoundError, S3ObjectStore


def test_key_layout():
    assert storage.result_key("EXP-005", "s4__krum__pm__seed42") == "sweeps/EXP-005/results/s4__krum__pm__seed42.json"
    assert storage.signal_key("EXP-005", "u") == "sweeps/EXP-005/signals/u.jsonl"
    assert storage.marker_key("EXP-005", "u") == "sweeps/EXP-005/done/u.marker"
    assert storage.manifest_key("EXP-005") == "sweeps/EXP-005/manifest.json"


def test_in_memory_put_head_get():
    s = InMemoryObjectStore()
    assert s.head("k") is False
    s.put_bytes("k", b"hello")
    assert s.head("k") is True
    assert s.get_bytes("k") == b"hello"


def test_in_memory_conditional_put_rejects_existing():
    s = InMemoryObjectStore()
    s.put_bytes("m", b"", if_none_match=True)
    with pytest.raises(ObjectExistsError):
        s.put_bytes("m", b"", if_none_match=True)


def test_in_memory_get_bytes_missing_key_raises():
    s = InMemoryObjectStore()
    with pytest.raises(ObjectNotFoundError):
        s.get_bytes("absent")


def test_in_memory_put_file(tmp_path):
    s = InMemoryObjectStore()
    p = tmp_path / "r.json"
    p.write_text('{"ok": true}')
    s.put_file("sweeps/E/results/u.json", p)
    assert s.get_bytes("sweeps/E/results/u.json") == b'{"ok": true}'


# ---------------------------------------------------------------------------
# S3ObjectStore tests (dependency-injected boto3 client via MagicMock)
# ---------------------------------------------------------------------------

def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "op")


def test_s3_put_bytes_conditional_maps_precondition_to_object_exists():
    client = MagicMock()
    client.put_object.side_effect = _client_error("PreconditionFailed")
    store = S3ObjectStore("bucket", client)
    with pytest.raises(ObjectExistsError):
        store.put_bytes("sweeps/E/done/u.marker", b"", if_none_match=True)
    call = client.put_object.call_args
    assert call.kwargs["IfNoneMatch"] == "*"
    assert call.kwargs["Bucket"] == "bucket"
    assert call.kwargs["Key"] == "sweeps/E/done/u.marker"


def test_s3_head_returns_false_on_404():
    client = MagicMock()
    client.head_object.side_effect = _client_error("404")
    assert S3ObjectStore("bucket", client).head("missing") is False


def test_s3_head_true_when_object_present():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 3}
    assert S3ObjectStore("bucket", client).head("present") is True


def test_s3_get_bytes_maps_nosuchkey_to_object_not_found():
    client = MagicMock()
    client.get_object.side_effect = _client_error("NoSuchKey")
    with pytest.raises(ObjectNotFoundError):
        S3ObjectStore("bucket", client).get_bytes("absent")


def test_s3_get_bytes_returns_body_bytes():
    client = MagicMock()
    client.get_object.return_value = {"Body": MagicMock(read=lambda: b"payload")}
    assert S3ObjectStore("bucket", client).get_bytes("k") == b"payload"


def test_s3_put_file_calls_upload_file_with_correct_args(tmp_path):
    client = MagicMock()
    p = tmp_path / "artifact.json"; p.write_bytes(b"data")
    S3ObjectStore("my-bucket", client).put_file("sweeps/E/results/u.json", p)
    client.upload_file.assert_called_once_with(str(p), "my-bucket", "sweeps/E/results/u.json")


def test_s3_put_bytes_reraises_non_precondition_clienterror():
    client = MagicMock()
    client.put_object.side_effect = _client_error("AccessDenied")
    with pytest.raises(ClientError):
        S3ObjectStore("bucket", client).put_bytes("k", b"x")


def test_in_memory_size_returns_length_or_none():
    s = InMemoryObjectStore()
    assert s.size("absent") is None
    s.put_bytes("k", b"hello")
    assert s.size("k") == 5


def test_s3_size_returns_content_length_and_none_on_missing():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 42}
    assert S3ObjectStore("b", client).size("k") == 42
    client.head_object.side_effect = _client_error("404")
    assert S3ObjectStore("b", client).size("missing") is None


def test_in_memory_delete_removes_key_and_absent_key_is_noop():
    """ : launch rollback deletes the manifest
    this launch wrote; delete must be idempotent (absent key = no-op)."""
    s = InMemoryObjectStore()
    s.put_bytes("sweeps/E/manifest.json", b"{}")
    s.delete("sweeps/E/manifest.json")
    assert s.head("sweeps/E/manifest.json") is False
    s.delete("sweeps/E/manifest.json")  # no-op, must not raise


def test_s3_delete_calls_delete_object():
    client = MagicMock()
    S3ObjectStore("my-bucket", client).delete("sweeps/E/manifest.json")
    client.delete_object.assert_called_once_with(
        Bucket="my-bucket", Key="sweeps/E/manifest.json"
    )


def test_in_memory_find_keys_prefix_scan_with_limit():
    """ : namespace-emptiness probe — the
    launch freshness check must be prefix-level, not unit-id-keyed."""
    s = InMemoryObjectStore()
    s.put_bytes("sweeps/EXP-005/done/u1.marker", b"")
    s.put_bytes("sweeps/EXP-005/results/u1.json", b"{}")
    s.put_bytes("sweeps/EXP-0055/manifest.json", b"{}")  # different namespace
    assert s.find_keys("sweeps/EXP-005/") == [
        "sweeps/EXP-005/done/u1.marker",
        "sweeps/EXP-005/results/u1.json",
    ]
    assert s.find_keys("sweeps/EXP-005/", limit=1) == ["sweeps/EXP-005/done/u1.marker"]
    assert s.find_keys("sweeps/EXP-999/") == []


def test_s3_find_keys_calls_list_objects_v2():
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": [{"Key": "sweeps/E/done/u1.marker"}, {"Key": "sweeps/E/results/u1.json"}],
    }
    keys = S3ObjectStore("my-bucket", client).find_keys("sweeps/E/", limit=25)
    client.list_objects_v2.assert_called_once_with(
        Bucket="my-bucket", Prefix="sweeps/E/", MaxKeys=25
    )
    assert keys == ["sweeps/E/done/u1.marker", "sweeps/E/results/u1.json"]


def test_s3_find_keys_empty_when_no_contents():
    client = MagicMock()
    client.list_objects_v2.return_value = {"KeyCount": 0}
    assert S3ObjectStore("b", client).find_keys("sweeps/E/") == []
