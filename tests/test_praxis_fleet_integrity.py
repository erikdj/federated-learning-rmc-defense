import pytest
from praxis_exp import storage
from praxis_exp.storage import InMemoryObjectStore
from praxis_exp.integrity import persist_unit, is_done, IntegrityError


def _files(tmp_path):
    r = tmp_path / "r.json"; r.write_text('{"return_code": 0}')
    s = tmp_path / "s.jsonl"; s.write_text('{"row": 1}\n')
    return r, s


def test_persist_writes_all_three_keys_with_marker_present_last(tmp_path):
    store = InMemoryObjectStore()
    r, s = _files(tmp_path)
    persist_unit(store, "EXP-005", "u", r, s)
    assert store.head(storage.result_key("EXP-005", "u"))
    assert store.head(storage.signal_key("EXP-005", "u"))
    assert store.head(storage.marker_key("EXP-005", "u"))


def test_marker_is_written_after_payload(tmp_path):
    written: list[str] = []
    store = InMemoryObjectStore()
    orig_put_bytes, orig_put_file = store.put_bytes, store.put_file
    store.put_bytes = lambda k, d, **kw: (written.append(k), orig_put_bytes(k, d, **kw))[1]
    store.put_file = lambda k, p: (written.append(k), orig_put_file(k, p))[1]
    r, s = _files(tmp_path)
    persist_unit(store, "EXP-005", "u", r, s)
    assert written[-1] == storage.marker_key("EXP-005", "u")
    assert written.index(storage.result_key("EXP-005", "u")) < written.index(storage.marker_key("EXP-005", "u"))
    assert written.index(storage.signal_key("EXP-005", "u")) < written.index(storage.marker_key("EXP-005", "u"))


def test_verify_after_write_raises_if_payload_absent(tmp_path):
    class DropResultStore(InMemoryObjectStore):
        def put_file(self, key, path):
            if key.endswith(".json"):
                return  # silently drop the result upload
            super().put_file(key, path)

    r, s = _files(tmp_path)
    with pytest.raises(IntegrityError):
        persist_unit(DropResultStore(), "EXP-005", "u", r, s)


def test_persist_is_idempotent_when_marker_exists(tmp_path):
    store = InMemoryObjectStore()
    r, s = _files(tmp_path)
    persist_unit(store, "EXP-005", "u", r, s)
    persist_unit(store, "EXP-005", "u", r, s)   # second call must not raise
    assert is_done(store, "EXP-005", "u")


def test_idempotent_rerun_preserves_payload_content(tmp_path):
    store = InMemoryObjectStore()
    r, s = _files(tmp_path)
    persist_unit(store, "EXP-005", "u", r, s)
    persist_unit(store, "EXP-005", "u", r, s)
    assert store.get_bytes(storage.result_key("EXP-005", "u")) == r.read_bytes()
    assert store.get_bytes(storage.signal_key("EXP-005", "u")) == s.read_bytes()


def test_verify_raises_when_signal_dropped(tmp_path):
    class DropSignalStore(InMemoryObjectStore):
        def put_file(self, key, path):
            if key.endswith(".jsonl"):
                return
            super().put_file(key, path)
    r, s = _files(tmp_path)
    with pytest.raises(IntegrityError):
        persist_unit(DropSignalStore(), "EXP-005", "u", r, s)


def test_verify_raises_on_size_mismatch(tmp_path):
    class TruncatingStore(InMemoryObjectStore):
        def size(self, key):
            real = super().size(key)
            return None if real is None else max(real - 1, 0)  # report 1 byte short
    r, s = _files(tmp_path)
    with pytest.raises(IntegrityError, match="size mismatch"):
        persist_unit(TruncatingStore(), "EXP-005", "u", r, s)


def test_preflight_raises_on_missing_source_file(tmp_path):
    store = InMemoryObjectStore()
    missing = tmp_path / "nope.json"
    s = tmp_path / "s.jsonl"; s.write_text("{}\n")
    with pytest.raises(IntegrityError, match="result_path"):
        persist_unit(store, "EXP-005", "u", missing, s)
