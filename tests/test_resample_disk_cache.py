"""Node-local disk resample cache + driver prewarm (image v9).

v8 shipped a per-PROCESS in-RAM LRU resample cache. At fleet shape (Ray
ActorPool: 32 actors > 21 concurrent clients, no client->actor affinity) that
cache measured a ~16% hit rate and never warmed up (see
tests/test_resample_cache_fleet_shape.py and tests/test_resample_cache_fleet_shape.py).
v9 replaces it with a NODE-LOCAL DISK cache shared by every actor process on the
container, primed once by the driver before run_simulation().

This module pins the correctness contract of the disk layer:
  * a disk-cached load is BYTE-IDENTICAL to a fresh resample and to the L1 result
    (the cache is a pure performance layer over a deterministic function),
  * the SMOTE-off path never touches either cache layer,
  * a torn/partial .npz is treated as a miss (recompute), never a crash,
  * the driver prewarm emits the per-client [SMOTE] provenance records to the
    driver's OWN stdout (which reaches CloudWatch under log_to_driver=False) AND
    returns them so the captured-stdout parser sees exactly one record/client.
"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import flowerfl.task as task_module
from flowerfl.task import load_data

TMP_DATASET = "disk_cache_ds"


@pytest.fixture
def isolated_cache_dir(tmp_path, monkeypatch):
    """Point the node-local disk cache at a fresh per-test directory."""
    d = tmp_path / "resample_cache"
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_DIR", str(d))
    task_module._reset_resample_cache()
    yield d
    task_module._reset_resample_cache()


@pytest.fixture
def skewed_dataset(tmp_path, monkeypatch, isolated_cache_dir):
    """A temp DATASET_CONFIGS entry backed by skewed parquet partitions.

    Three client files so the prewarm test can iterate a 3-supernode federation
    with a 1:1 partition->client mapping (no client_idx aliasing -> no spurious
    provenance conflicts)."""
    rng = np.random.default_rng(0)
    n_maj, n_min, n_feat = 300, 90, 6

    def _frame(shift):
        X = np.vstack([rng.normal(0.0, 1.0, (n_maj, n_feat)),
                       rng.normal(shift, 0.4, (n_min, n_feat))])
        y = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)
        cols = {f"f{i}": X[:, i] for i in range(n_feat)}
        cols["Attack_label"] = y
        return pd.DataFrame(cols)

    data_dir = tmp_path / "disk_ds"
    data_dir.mkdir()
    files = ["client_0.parquet", "client_1.parquet", "client_2.parquet"]
    for i, f in enumerate(files):
        _frame(3.0 + i).to_parquet(data_dir / f)

    cfg = dict(task_module.DATASET_CONFIGS)
    cfg[TMP_DATASET] = {
        "data_dir": str(data_dir),
        "label_column": "Attack_label",
        "client_files": files,
        "client_ids": ["0", "1", "2"],
        "num_classes": 2,
        "description": "temp skewed partitions for disk-cache tests",
        "malicious_order": [0, 1, 2],
    }
    monkeypatch.setattr(task_module, "DATASET_CONFIGS", cfg)
    task_module._input_shape_cache.pop(TMP_DATASET, None)
    yield TMP_DATASET


def _train_rows(loader):
    ds = loader.dataset
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([torch.as_tensor(ds[i][1]) for i in range(len(ds))])
    return X, y


def _base(**over):
    b = dict(
        dataset_name=TMP_DATASET, batch_size=32, train_split=0.8, val_split=0.1,
        smote_enabled=True, smote_variant="smote", smote_target="balanced",
        smote_seed=42,
    )
    b.update(over)
    return b


# ---------------------------------------------------------------------------
# 1. Bit-identity: disk-cached load == fresh resample == v8-L1-cached result
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant,target", [
    ("smote", "balanced"),
    ("random_under", "balanced"),
])
def test_disk_load_bit_identical_to_fresh_and_l1(skewed_dataset, variant, target):
    kw = _base(smote_variant=variant, smote_target=target)

    # (a) cold miss -> computes, writes disk, fills L1. Rows R1 come from L1.
    tr1, _, _ = load_data(0, **kw)
    assert task_module._resample_cache_misses == 1
    assert task_module._resample_cache_hits == 0
    X1, y1 = _train_rows(tr1)
    # the disk artifact exists
    npz = list(Path(list(task_module._resample_cache_dir() for _ in [0])[0]).glob("*.npz"))
    assert npz, "miss did not write a disk cache file"

    # (b) drop L1 ONLY (disk survives) -> next load is a DISK hit. Rows R2.
    task_module._resample_cache.clear()
    tr2, _, _ = load_data(0, **kw)
    assert task_module._resample_cache_hits == 1, "disk layer did not serve the hit"
    assert task_module._resample_cache_misses == 1, "disk hit must not recompute"
    X2, y2 = _train_rows(tr2)

    # (c) wipe BOTH layers -> fresh recompute from scratch. Rows R3.
    task_module._reset_resample_cache()
    tr3, _, _ = load_data(0, **kw)
    assert task_module._resample_cache_misses == 1
    X3, y3 = _train_rows(tr3)

    # disk-load == L1 == fresh, byte-identical
    assert torch.equal(X1, X2) and torch.equal(y1, y2)
    assert torch.equal(X1, X3) and torch.equal(y1, y3)


def test_off_path_touches_neither_layer(skewed_dataset):
    load_data(0, dataset_name=TMP_DATASET, batch_size=32, smote_enabled=False)
    assert len(task_module._resample_cache) == 0
    assert task_module._resample_cache_hits == 0
    assert task_module._resample_cache_misses == 0
    assert not list(Path(task_module._resample_cache_dir()).glob("*.npz")), \
        "OFF path must never write a disk cache file"


# ---------------------------------------------------------------------------
# 2. Atomicity / robustness: a torn .npz is a miss (recompute), never a crash
# ---------------------------------------------------------------------------
def test_torn_disk_file_is_treated_as_miss(skewed_dataset):
    kw = _base()
    # prime a valid entry, capture its rows
    tr_ref, _, _ = load_data(0, **kw)
    Xref, yref = _train_rows(tr_ref)

    # corrupt the on-disk artifact into a torn file, drop L1
    cache_dir = Path(task_module._resample_cache_dir())
    npz = list(cache_dir.glob("*.npz"))[0]
    npz.write_bytes(b"\x00\x01torn-not-an-npz\x02")
    task_module._reset_resample_cache()
    # rewrite the corrupt file AFTER reset wiped it (reset clears disk too)
    npz.write_bytes(b"\x00\x01torn-not-an-npz\x02")

    # load must recompute (miss), not raise, and produce the same rows
    tr2, _, _ = load_data(0, **kw)
    assert task_module._resample_cache_misses == 1, "torn file should force a recompute"
    X2, y2 = _train_rows(tr2)
    assert torch.equal(Xref, X2) and torch.equal(yref, y2)


def test_disk_write_is_atomic_no_partial_files_left(skewed_dataset):
    load_data(0, **_base())
    cache_dir = Path(task_module._resample_cache_dir())
    leftovers = [p.name for p in cache_dir.iterdir()
                 if p.name.startswith(task_module._RESAMPLE_DISK_TMP_PREFIX)]
    assert leftovers == [], f"atomic write left temp files behind: {leftovers}"


# ---------------------------------------------------------------------------
# 3. Driver prewarm: emits [SMOTE] records to driver stdout AND to the parser
# ---------------------------------------------------------------------------
def test_prewarm_populates_disk_and_emits_parseable_records(skewed_dataset, monkeypatch, capsys):
    import run_phase4_flower as runner

    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    run_config = {
        "dataset": TMP_DATASET,
        "batch-size": 32,
        "max-samples": 0,
        "seed": 42,
        "smote-enabled": True,
        "smote-variant": "smote",
        "smote-target": "balanced",
    }

    text, summary = runner._prewarm_resample_cache(run_config)

    # one record per supernode, all applied, no conflicts
    assert summary["smote_enabled"] is True
    assert summary["prewarmed_clients"] == 3
    records = runner.parse_smote_records(text)
    run_summary = runner._smote_run_summary(records)
    assert run_summary["smote_applied_count"] == 3
    assert run_summary["smote_record_conflicts"] == 0

    # the disk cache was populated (workers will read, not recompute)
    assert len(list(Path(task_module._resample_cache_dir()).glob("*.npz"))) == 3

    # provenance reached the DRIVER's real stdout (CloudWatch under
    # log_to_driver=False) — the v8 observability gap this fixes
    out = capsys.readouterr().out
    assert "[SMOTE]" in out
    assert "[PREWARM]" in out


def test_prewarm_off_is_noop(skewed_dataset, monkeypatch, capsys):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    text, summary = runner._prewarm_resample_cache({
        "dataset": TMP_DATASET, "smote-enabled": False,
    })
    assert text == ""
    assert summary["smote_enabled"] is False
    assert summary["prewarmed_clients"] == 0
    assert not list(Path(task_module._resample_cache_dir()).glob("*.npz"))
    assert "[SMOTE]" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 4. provenance must NOT vanish on cache HITS. A second
#    invocation against a pre-populated cache (or a later config in a
#    multi-config run) still emits the full per-client record set — otherwise it
#    reports smote_applied_count=0 while training on resampled data.
# ---------------------------------------------------------------------------
def test_prewarm_emits_full_record_set_even_when_all_hits(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    rc = {
        "dataset": TMP_DATASET, "batch-size": 32, "max-samples": 0, "seed": 42,
        "smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced",
    }

    text_miss, _ = runner._prewarm_resample_cache(rc)   # first: all MISSES
    text_hit, _ = runner._prewarm_resample_cache(rc)    # second: all HITS

    recs_miss = runner.parse_smote_records(text_miss)
    recs_hit = runner.parse_smote_records(text_hit)

    # THE fix: the all-hits invocation still reports every client as applied.
    assert runner._smote_run_summary(recs_miss)["smote_applied_count"] == 3
    assert runner._smote_run_summary(recs_hit)["smote_applied_count"] == 3
    assert runner._smote_run_summary(recs_hit)["smote_record_conflicts"] == 0

    # identical records modulo the mutable cache_hits observability token
    keys_miss = sorted(runner._smote_dedupe_key(r) for r in recs_miss)
    keys_hit = sorted(runner._smote_dedupe_key(r) for r in recs_hit)
    assert keys_miss == keys_hit


# ---------------------------------------------------------------------------
# 5. a disk-store failure (ENOSPC / read-only fs) must NOT
#    lose the computed resample. Persistence is best-effort — load_data returns
#    the REAL resampled arrays + a loud warning, never falling through the broad
#    handler into generate_synthetic_data (which would silently substitute
#    synthetic data for the client's real partition).
# ---------------------------------------------------------------------------
def test_disk_store_failure_returns_real_arrays_not_synthetic(skewed_dataset, monkeypatch, capsys):
    kw = _base()
    tr_ref, _, _ = load_data(0, **kw)
    Xref, yref = _train_rows(tr_ref)
    task_module._reset_resample_cache()

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(task_module, "_resample_disk_store", _enospc)
    tr, _, _ = load_data(0, **kw)
    X, y = _train_rows(tr)

    out = capsys.readouterr().out
    assert "Generating synthetic data" not in out, "store failure fell into synthetic fallback"
    assert "[RESAMPLE-CACHE]" in out and "WARNING" in out and "errno=28" in out
    assert torch.equal(Xref, X) and torch.equal(yref, y), "did not return the real resample"
    assert task_module._resample_cache_misses == 1


# ---------------------------------------------------------------------------
# 6. corrupt cache entries must NOT be immortal. A rejected
#    (torn/legacy) .npz is overwritten on recompute — not discarded by the
#    lost-a-race guard — so it self-heals in one call instead of recomputing
#    forever.
# ---------------------------------------------------------------------------
def test_corrupt_disk_entry_self_heals(skewed_dataset):
    kw = _base()
    load_data(0, **kw)  # prime a valid entry
    cache_dir = Path(task_module._resample_cache_dir())
    npz = list(cache_dir.glob("*.npz"))[0]

    # corrupt the on-disk entry, drop L1 (keep the corrupt file on disk)
    task_module._resample_cache.clear()
    npz.write_bytes(b"\x00 torn not-an-npz \x01")
    assert task_module._resample_disk_load(str(npz)) is None, "test precondition: file unreadable"

    # ONE call self-heals: reject -> recompute -> overwrite the corrupt bytes
    load_data(0, **kw)
    assert task_module._resample_disk_load(str(npz)) is not None, "corrupt entry not healed"

    # a subsequent call now HITS the healed entry (no perpetual recompute)
    task_module._resample_cache.clear()
    misses_before = task_module._resample_cache_misses
    hits_before = task_module._resample_cache_hits
    load_data(0, **kw)
    assert task_module._resample_cache_hits == hits_before + 1
    assert task_module._resample_cache_misses == misses_before


# ---------------------------------------------------------------------------
# 7. the disk layer is bounded by a byte budget — evict oldest
#    mtime before each store so a multi-seed run cannot grow /tmp without limit.
# ---------------------------------------------------------------------------
def _write_entry(cache_dir, name, nbytes, mtime_ns):
    p = os.path.join(str(cache_dir), f"{name}.npz")
    with open(p, "wb") as fh:
        fh.write(b"x" * nbytes)
    os.utime(p, ns=(mtime_ns, mtime_ns))
    return p


def test_evict_for_budget_removes_oldest_mtime(isolated_cache_dir, monkeypatch):
    d = isolated_cache_dir
    d.mkdir(parents=True, exist_ok=True)
    a = _write_entry(d, "a", 100, 1_000)   # oldest
    b = _write_entry(d, "b", 100, 2_000)   # newest
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", "250")

    # total 200 + incoming 100 = 300 > 250 -> evict exactly the oldest (a) -> 200
    fits = task_module._evict_for_budget(str(d), 100)
    assert fits is True, "incoming should fit after evicting the oldest entry"
    assert not os.path.exists(a), "oldest-mtime entry was not evicted"
    assert os.path.exists(b), "newer entry was wrongly evicted"
    assert task_module._resample_disk_evictions == 1


def test_evict_tolerates_concurrent_deletion(isolated_cache_dir, monkeypatch):
    d = isolated_cache_dir
    d.mkdir(parents=True, exist_ok=True)
    _write_entry(d, "a", 100, 1_000)
    _write_entry(d, "b", 100, 2_000)
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", "150")

    real_unlink = os.unlink
    calls = {"n": 0}

    def flaky_unlink(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError(2, "raced: already evicted", p)
        return real_unlink(p)

    monkeypatch.setattr(os, "unlink", flaky_unlink)
    # must NOT raise despite the racing ENOENT on the first eviction target
    task_module._evict_for_budget(str(d), 100)


def test_load_data_respects_disk_budget(skewed_dataset, monkeypatch):
    load_data(0, **_base(smote_seed=1))
    entry = list(Path(task_module._resample_cache_dir()).glob("*.npz"))[0]
    s = entry.stat().st_size
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(int(2.5 * s)))

    for seed in (2, 3, 4, 5):
        task_module._resample_cache.clear()  # force the disk path each time
        load_data(0, **_base(smote_seed=seed))

    files = list(Path(task_module._resample_cache_dir()).glob("*.npz"))
    total = sum(p.stat().st_size for p in files)
    assert total <= int(2.5 * s) + s, "disk cache exceeded its byte budget"
    assert len(files) < 5, "no eviction happened — cache grew unbounded"
    assert task_module._resample_disk_evictions >= 1


# ---------------------------------------------------------------------------
# 8. the key is bound to source-data identity + format version,
#    so a regenerated parquet (or a serialization change) can never serve stale
#    training rows against fresh val/test rows.
# ---------------------------------------------------------------------------
def test_source_parquet_change_invalidates_entry(skewed_dataset):
    kw = _base()
    load_data(0, **kw)
    assert task_module._resample_cache_misses == 1

    cfg = task_module.DATASET_CONFIGS[TMP_DATASET]
    fpath = os.path.join(cfg["data_dir"], cfg["client_files"][0])
    st = os.stat(fpath)
    bumped = st.st_mtime_ns + 1_000_000_000
    os.utime(fpath, ns=(bumped, bumped))  # simulate regeneration (mtime changes)

    task_module._resample_cache.clear()  # drop L1 so the disk key is consulted
    load_data(0, **kw)
    assert task_module._resample_cache_misses == 2, \
        "a regenerated parquet served a STALE cached resample"


def test_cache_format_version_bump_invalidates(skewed_dataset, monkeypatch):
    kw = _base()
    load_data(0, **kw)
    assert task_module._resample_cache_misses == 1

    monkeypatch.setattr(task_module, "_RESAMPLE_CACHE_FORMAT_VERSION", 999)
    task_module._resample_cache.clear()
    load_data(0, **kw)
    assert task_module._resample_cache_misses == 2, \
        "a CACHE_FORMAT_VERSION bump did not invalidate the prior entry"


# ---------------------------------------------------------------------------
# 9. cache-DIRECTORY setup is best-effort. An uncreatable
#    PRAXIS_RESAMPLE_CACHE_DIR must never raise — not at prewarm (which would
#    abort an enabled-SMOTE run) nor on the worker path (which would escape into
#    load_data's broad handler -> synthetic-data substitution).
# ---------------------------------------------------------------------------
def test_resample_cache_dir_best_effort_returns_none_and_warns_once(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "iam_a_file"
    blocker.write_text("x")  # a FILE where a dir parent is expected -> makedirs fails
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_DIR", str(blocker / "cache"))
    task_module._resample_disk_disabled_dirs.clear()

    assert task_module._resample_cache_dir() is None
    assert task_module._resample_cache_dir() is None  # second call also never raises
    out = capsys.readouterr().out
    assert out.count("DISK CACHE DISABLED") == 1, "should warn exactly once per dir"


def test_load_data_uncreatable_cache_dir_returns_real_arrays(skewed_dataset, monkeypatch, tmp_path, capsys):
    kw = _base()
    ref_tr, _, _ = load_data(0, **kw)          # healthy run, good dir (fixture)
    Xref, yref = _train_rows(ref_tr)

    blocker = tmp_path / "blocker_file"
    blocker.write_text("x")
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_DIR", str(blocker / "cache"))
    task_module._resample_disk_disabled_dirs.clear()
    task_module._resample_cache.clear()        # force the compute path (no L1)

    tr, _, _ = load_data(0, **kw)
    X, y = _train_rows(tr)
    out = capsys.readouterr().out
    assert "Generating synthetic data" not in out, "uncreatable dir fell into synthetic fallback"
    assert torch.equal(Xref, X) and torch.equal(yref, y), "did not return the real resample"
    assert task_module._resample_compute_only_count >= 1


def test_prewarm_survives_uncreatable_cache_dir(skewed_dataset, monkeypatch, tmp_path, capsys):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    blocker = tmp_path / "prewarm_blocker"
    blocker.write_text("x")
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_DIR", str(blocker / "cache"))
    task_module._resample_disk_disabled_dirs.clear()

    rc = {
        "dataset": TMP_DATASET, "batch-size": 32, "max-samples": 0, "seed": 42,
        "smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced",
    }
    text, summary = runner._prewarm_resample_cache(rc)

    # provenance intact despite no disk layer
    assert runner._smote_run_summary(runner.parse_smote_records(text))["smote_applied_count"] == 3
    assert summary["compute_only"] == 3
    assert summary["persisted"] == 0
    out = capsys.readouterr().out
    assert "DISK CACHE DISABLED" in out
    assert "[PREWARM]" in out and "compute_only=3" in out


# ---------------------------------------------------------------------------
# 10. the prewarm surfaces persisted vs compute_only and
#     re-emits [RESAMPLE-CACHE] warnings, so a degraded prime (failed stores) is
#     unmistakable in CloudWatch instead of falsely reporting every client primed.
# ---------------------------------------------------------------------------
def test_prewarm_reports_compute_only_and_warnings_on_store_failure(skewed_dataset, monkeypatch, capsys):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(task_module, "_resample_disk_store", _enospc)
    rc = {
        "dataset": TMP_DATASET, "batch-size": 32, "max-samples": 0, "seed": 42,
        "smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced",
    }
    text, summary = runner._prewarm_resample_cache(rc)

    # provenance still complete, but the summary shows the degradation
    assert runner._smote_run_summary(runner.parse_smote_records(text))["smote_applied_count"] == 3
    assert summary["compute_only"] == 3
    assert summary["persisted"] == 0
    out = capsys.readouterr().out
    assert "[RESAMPLE-CACHE] WARNING disk store failed" in out, "store-failure warning was swallowed"
    assert "compute_only=3" in out


def test_prewarm_healthy_reports_all_persisted(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    rc = {
        "dataset": TMP_DATASET, "batch-size": 32, "max-samples": 0, "seed": 42,
        "smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced",
    }
    _, summary = runner._prewarm_resample_cache(rc)
    assert summary["persisted"] == 3
    assert summary["compute_only"] == 0


# ---------------------------------------------------------------------------
# 11. an L1 hit is only "persisted" if the L2 .npz still exists
#     (byte-budget eviction may have removed it). It must be verified and, if
#     gone, restored — otherwise the summary claims durable while Ray actors
#     (who cannot see the driver L1) recompute that partition forever.
# ---------------------------------------------------------------------------
_RC3 = {
    "dataset": TMP_DATASET, "batch-size": 32, "max-samples": 0, "seed": 42,
    "smote-enabled": True, "smote-variant": "smote", "smote-target": "balanced",
}


def test_prewarm_restores_evicted_disk_entry_on_l1_hit(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    runner._prewarm_resample_cache(_RC3)  # populate L1 (<=4) + 3 disk entries
    cache_dir = Path(task_module._resample_cache_dir())
    for p in cache_dir.glob("*.npz"):  # simulate byte-budget eviction of the .npz
        p.unlink()
    assert list(cache_dir.glob("*.npz")) == []

    # second prewarm: all L1 hits, but disk is empty -> must restore + persist
    _, summary = runner._prewarm_resample_cache(_RC3)
    assert summary["persisted"] == 3, "evicted disk entry not restored/persisted on L1 hit"
    assert summary["compute_only"] == 0
    assert len(list(cache_dir.glob("*.npz"))) == 3, "disk entries were not re-created"


def test_prewarm_l1_hit_restore_failure_counts_compute_only(skewed_dataset, monkeypatch, capsys):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    runner._prewarm_resample_cache(_RC3)  # populate L1 + disk
    cache_dir = Path(task_module._resample_cache_dir())
    for p in cache_dir.glob("*.npz"):
        p.unlink()

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(task_module, "_resample_disk_store", _enospc)
    _, summary = runner._prewarm_resample_cache(_RC3)  # L1 hits, restore fails
    assert summary["persisted"] == 0
    assert summary["compute_only"] == 3
    assert "[RESAMPLE-CACHE] WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 12. an artifact larger than the budget (small-scratch config)
#     must SKIP persistence, never blow past the bound — but load_data still
#     returns the real resampled arrays.
# ---------------------------------------------------------------------------
def test_store_skipped_when_artifact_exceeds_budget(skewed_dataset, monkeypatch, capsys):
    kw = _base()
    ref_tr, _, _ = load_data(0, **kw)             # reference under default budget
    Xref, yref = _train_rows(ref_tr)
    task_module._reset_resample_cache()           # wipe disk + L1

    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", "10")  # smaller than any artifact
    tr, _, _ = load_data(0, **kw)
    X, y = _train_rows(tr)

    out = capsys.readouterr().out
    cache_dir = Path(task_module._resample_cache_dir())
    assert list(cache_dir.glob("*.npz")) == [], "over-budget artifact was persisted anyway"
    assert "does not fit cache budget" in out, "no over-budget skip warning emitted"
    assert task_module._resample_compute_only_count >= 1
    # bound respected AND correctness preserved
    total = sum(p.stat().st_size for p in cache_dir.glob("*.npz"))
    assert total <= 10
    assert torch.equal(Xref, X) and torch.equal(yref, y), "did not return the real resample"


# ---------------------------------------------------------------------------
# 13. a cache READ failure of ANY kind (incl. zipfile.BadZipFile
#     from a damaged ZIP central directory) is a miss — it must never propagate
#     into load_data's broad handler and substitute synthetic data.
# ---------------------------------------------------------------------------
def test_badzipfile_cache_entry_treated_as_miss_and_healed(skewed_dataset, capsys):
    kw = _base()
    ref_tr, _, _ = load_data(0, **kw)
    Xref, yref = _train_rows(ref_tr)

    cache_dir = Path(task_module._resample_cache_dir())
    npz = list(cache_dir.glob("*.npz"))[0]
    # Corrupt the ZIP central directory: overwrite the tail (EOCD region) with
    # garbage while leaving the leading PK magic, so np.load routes to the zip
    # path and raises zipfile.BadZipFile.
    data = bytearray(npz.read_bytes())
    for i in range(1, min(40, len(data)) + 1):
        data[-i] = 0xFF
    npz.write_bytes(bytes(data))
    assert task_module._resample_disk_load(str(npz)) is None, "corrupt zip should read as miss"

    task_module._resample_cache.clear()  # force the disk consult (keep corrupt file)
    capsys.readouterr()                   # drain
    tr, _, _ = load_data(0, **kw)
    X, y = _train_rows(tr)

    out = capsys.readouterr().out
    assert "Generating synthetic data" not in out, "damaged cache file fell into synthetic fallback"
    assert "unreadable entry" in out, "no unreadable-entry warning emitted"
    assert torch.equal(Xref, X) and torch.equal(yref, y), "did not return the real resample"
    assert task_module._resample_disk_load(str(npz)) is not None, "entry was not self-healed"


# ---------------------------------------------------------------------------
# 14. evict+size-check+publish is serialized behind a best-effort
#     interprocess lock, so concurrent stores can never each snapshot the same
#     total and all publish (bound exceeded). Lock-unavailable => skip, not block.
# ---------------------------------------------------------------------------
def _race_store_worker(cache_dir, max_bytes, idx, barrier):
    os.environ["PRAXIS_RESAMPLE_CACHE_DIR"] = cache_dir
    os.environ["PRAXIS_RESAMPLE_CACHE_MAX_BYTES"] = max_bytes
    import numpy as _np
    import flowerfl.task as _task
    X = _np.full((2000, 8), float(idx), dtype=_np.float32)
    y = _np.zeros(2000, dtype=_np.int64)
    path = _task._resample_disk_path(("race", idx))
    try:
        barrier.wait(timeout=15)  # maximize overlap on the store
    except Exception:
        pass
    _task._resample_disk_store(path, X, y, None)


def test_concurrent_stores_never_exceed_budget(tmp_path):
    import multiprocessing as mp
    if not hasattr(task_module, "fcntl") or task_module.fcntl is None:
        import pytest
        pytest.skip("fcntl unavailable — interprocess lock not testable here")

    cache_dir = tmp_path / "race_cache"
    cache_dir.mkdir()
    one = np.zeros((2000, 8), dtype=np.float32).nbytes + np.zeros(2000, dtype=np.int64).nbytes
    budget = int(one * 1.5)  # fits exactly ONE artifact, not two

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    procs = [
        ctx.Process(target=_race_store_worker, args=(str(cache_dir), str(budget), i, barrier))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    for p in procs:
        if p.is_alive():
            p.terminate()

    files = list(cache_dir.glob("*.npz"))
    total = sum(f.stat().st_size for f in files)
    assert total <= budget, f"budget bound exceeded under concurrency: {total} > {budget} ({len(files)} files)"
    assert len(files) <= 1


def test_store_skips_when_budget_lock_unavailable(skewed_dataset, monkeypatch, capsys):
    kw = _base()
    ref_tr, _, _ = load_data(0, **kw)
    Xref, yref = _train_rows(ref_tr)
    task_module._reset_resample_cache()

    monkeypatch.setattr(task_module, "_acquire_budget_lock", lambda cache_dir: None)
    tr, _, _ = load_data(0, **kw)
    X, y = _train_rows(tr)

    out = capsys.readouterr().out
    assert list(Path(task_module._resample_cache_dir()).glob("*.npz")) == [], \
        "store proceeded without the budget lock"
    assert "could not acquire budget lock" in out
    assert task_module._resample_compute_only_count >= 1
    assert torch.equal(Xref, X) and torch.equal(yref, y), "did not return the real resample"


# ---------------------------------------------------------------------------
# 15. under the exclusive budget lock, EVERY other temp is a
#     dead writer's orphan and is removed IMMEDIATELY (no age gate), so dead
#     gigabytes never charge against the budget.
# ---------------------------------------------------------------------------
def _mktemp(cache_dir, name, nbytes, mtime_ns):
    p = os.path.join(str(cache_dir), task_module._RESAMPLE_DISK_TMP_PREFIX + name + ".npz")
    with open(p, "wb") as fh:
        fh.write(b"x" * nbytes)
    os.utime(p, ns=(mtime_ns, mtime_ns))
    return p


def test_store_removes_all_orphan_temps_regardless_of_age(skewed_dataset):
    cache_dir = Path(task_module._resample_cache_dir())
    now = time.time_ns()
    old = _mktemp(cache_dir, "old", 100, now - 10 * 60 * 1_000_000_000)  # ~10 min old
    fresh = _mktemp(cache_dir, "fresh", 100, now)                        # fresh mtime

    # a real store runs cleanup under the budget lock — both orphans go
    load_data(0, **_base())

    assert not os.path.exists(old), "old orphan temp not removed"
    assert not os.path.exists(fresh), "fresh orphan temp not removed (stale age gate still present?)"
    assert task_module._resample_stale_temps_cleaned == 2


def test_evict_removes_fresh_orphan_temp_immediately(skewed_dataset):
    cache_dir = Path(task_module._resample_cache_dir())
    orphan = _mktemp(cache_dir, "orphan", 100, time.time_ns())  # fresh mtime
    before = task_module._resample_stale_temps_cleaned

    task_module._evict_for_budget(str(cache_dir), 100)

    assert not os.path.exists(orphan), "fresh orphan temp not removed under locked cleanup"
    assert task_module._resample_stale_temps_cleaned == before + 1


# ---------------------------------------------------------------------------
# 16. the prewarm health is persisted into result provenance
#     (a resample_prewarm block) so a degraded run is distinguishable in the
#     result JSON — not just ephemeral console output.
# ---------------------------------------------------------------------------
def test_prewarm_provenance_block_healthy(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    _, summary = runner._prewarm_resample_cache(_RC3)
    block = runner._resample_prewarm_provenance(summary)
    assert block is not None
    assert block["persisted"] == 3
    assert block["compute_only"] == 0
    assert "stale_temps_cleaned" in block
    assert block["budget_bytes"] == task_module._resample_cache_max_bytes()
    assert set(block) == {
        "persisted", "compute_only", "stores_succeeded", "evictions",
        "stale_temps_cleaned", "cache_bytes", "budget_bytes", "wall_clock_s",
    }


def test_prewarm_provenance_block_degraded(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(task_module, "_resample_disk_store", _enospc)
    _, summary = runner._prewarm_resample_cache(_RC3)
    block = runner._resample_prewarm_provenance(summary)
    assert block["compute_only"] == 3, "degraded prime not reflected in result provenance"
    assert block["persisted"] == 0


def test_prewarm_provenance_none_when_off():
    import run_phase4_flower as runner
    assert runner._resample_prewarm_provenance({"smote_enabled": False}) is None


# ---------------------------------------------------------------------------
# 17. a lost-race store (entry already published, non-overwrite)
#     returns persisted immediately WITHOUT running eviction — reserving budget
#     for a duplicate it would discard could evict unrelated HOT entries.
# ---------------------------------------------------------------------------
def test_lost_race_store_skips_eviction(skewed_dataset, monkeypatch):
    Xw = np.zeros((50, 4), dtype=np.float32)
    yw = np.zeros(50, dtype=np.int64)
    pw = task_module._resample_disk_path(("W", 0))  # the entry we lose the race on
    ph = task_module._resample_disk_path(("H", 0))  # an unrelated HOT entry
    assert task_module._resample_disk_store(pw, Xw, yw, None) is True
    assert task_module._resample_disk_store(ph, Xw, yw, None) is True

    # a budget so tight that eviction WOULD fire if the lost-race store reserved
    # space for its (discarded) duplicate
    one = Xw.nbytes + yw.nbytes
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(int(one * 1.5)))
    ev_before = task_module._resample_disk_evictions

    ret = task_module._resample_disk_store(pw, Xw, yw, None)  # lost race (pw exists)
    assert ret is True, "lost-race store should report persisted"
    assert task_module._resample_disk_evictions == ev_before, "lost-race store ran eviction"
    assert os.path.exists(ph), "lost-race store evicted an unrelated hot entry"
    assert os.path.exists(pw)


# ---------------------------------------------------------------------------
# 18. prewarm `persisted` reflects FINAL on-disk durability,
#     not the cumulative per-store counter — so same-loop eviction under a tight
#     budget is reported honestly (workers miss the evicted partitions).
# ---------------------------------------------------------------------------
def test_prewarm_persisted_reflects_final_durability(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    # size one artifact, then set a budget that can hold only ~one entry
    _, warm = runner._prewarm_resample_cache(_RC3)
    entry = list(Path(task_module._resample_cache_dir()).glob("*.npz"))[0]
    s = entry.stat().st_size
    task_module._reset_resample_cache()
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(int(s * 1.4)))

    _, summary = runner._prewarm_resample_cache(_RC3)
    # only the LAST-stored partition survives on disk; earlier ones were evicted
    assert summary["persisted"] == 1, f"expected 1 durable, got {summary['persisted']}"
    assert summary["compute_only"] == 2
    # the raw per-store counter still shows all three stores "succeeded"
    assert summary["stores_succeeded"] == 3
    assert len(list(Path(task_module._resample_cache_dir()).glob("*.npz"))) == 1


# ---------------------------------------------------------------------------
# 19. the budget decision uses the TRUE serialized on-disk size
#     (np.savez adds ZIP headers + the has_reason/reason arrays), not the raw
#     array nbytes — so an entry can never publish over the hard limit.
# ---------------------------------------------------------------------------
def test_budget_uses_true_serialized_size_not_nbytes(skewed_dataset, monkeypatch, capsys):
    X = np.zeros((100, 4), dtype=np.float32)
    y = np.zeros(100, dtype=np.int64)
    nbytes = X.nbytes + y.nbytes
    path = task_module._resample_disk_path(("edge8", 0))

    # budget == raw nbytes -> BELOW the true serialized size, so the store must
    # SKIP (a raw-nbytes estimate would have published slightly over-budget)
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(nbytes))
    assert task_module._resample_disk_store(path, X, y, None) is False
    assert not os.path.exists(path)
    assert "does not fit cache budget" in capsys.readouterr().out

    # with realistic headroom the same store publishes, within budget
    task_module._reset_resample_cache()
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(nbytes * 4))
    assert task_module._resample_disk_store(path, X, y, None) is True
    assert os.path.exists(path)
    assert os.path.getsize(path) <= nbytes * 4


# ---------------------------------------------------------------------------
# 20. the durability pass validates STRUCTURE, not existence;
#     and an un-repairable corrupt entry is REMOVED (missing > poisoned).
# ---------------------------------------------------------------------------
def test_resample_disk_valid_rejects_corrupt_and_missing(skewed_dataset):
    load_data(0, **_base())
    npz = list(Path(task_module._resample_cache_dir()).glob("*.npz"))[0]
    assert task_module._resample_disk_valid(str(npz)) is True
    npz.write_bytes(b"not a zip at all")
    assert task_module._resample_disk_valid(str(npz)) is False
    npz.unlink()
    assert task_module._resample_disk_valid(str(npz)) is False


def test_prewarm_corrupt_unrepairable_not_persisted_and_removed(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    runner._prewarm_resample_cache(_RC3)  # 3 valid entries + L1
    cache_dir = Path(task_module._resample_cache_dir())
    for p in cache_dir.glob("*.npz"):
        p.write_bytes(b"corrupt")       # poison every entry
    task_module._resample_cache.clear()  # force the disk consult

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(task_module, "_resample_disk_store", _enospc)
    _, summary = runner._prewarm_resample_cache(_RC3)  # repair fails for all

    assert summary["persisted"] == 0, "un-repairable corrupt entry counted persisted"
    assert summary["compute_only"] == 3
    assert list(cache_dir.glob("*.npz")) == [], "poisoned entries were not removed"


# ---------------------------------------------------------------------------
# 21. the prewarm enforces the CURRENT budget on a reused dir
#     ONCE at start, so a lowered budget can't stay oversized on all-hit loads.
# ---------------------------------------------------------------------------
def test_enforce_budget_evicts_reused_dir_down(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    runner._prewarm_resample_cache(_RC3)  # 3 entries under the default budget
    cache_dir = Path(task_module._resample_cache_dir())
    sizes = sorted(p.stat().st_size for p in cache_dir.glob("*.npz"))
    assert len(sizes) == 3
    budget = sizes[0] + sizes[1] + sizes[2] // 2  # holds ~2, not 3
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(budget))

    evicted = task_module.enforce_resample_disk_budget()
    total = sum(p.stat().st_size for p in cache_dir.glob("*.npz"))
    assert total <= budget, f"reused dir stayed oversized: {total} > {budget}"
    assert evicted >= 1


def test_prewarm_invokes_budget_enforcement(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)
    calls = []
    real = task_module.enforce_resample_disk_budget
    monkeypatch.setattr(
        task_module, "enforce_resample_disk_budget",
        lambda: (calls.append(1), real())[1],
    )
    runner._prewarm_resample_cache(_RC3)
    assert calls, "prewarm did not enforce the budget on the existing dir at start"


# ---------------------------------------------------------------------------
# 22. the r8 reorder wrote the full temp BEFORE eviction, which
#     np.savez-ENOSPCs on a near-full FS. The store pre-evicts for SPACE BEFORE
#     writing the temp. Assert the ORDERING via call-order recording (under a
#     realistic budget where the space pre-evict phase engages).
# ---------------------------------------------------------------------------
def test_store_pre_evicts_for_space_before_temp_write(skewed_dataset, monkeypatch):
    order = []
    real_evict = task_module._evict_for_budget
    real_savez = task_module.np.savez

    def spy_evict(*a, **k):
        order.append("evict")
        return real_evict(*a, **k)

    def spy_savez(*a, **k):
        order.append("savez")
        return real_savez(*a, **k)

    monkeypatch.setattr(task_module, "_evict_for_budget", spy_evict)
    monkeypatch.setattr(task_module.np, "savez", spy_savez)

    X = np.zeros((100, 4), dtype=np.float32)
    y = np.zeros(100, dtype=np.int64)
    assert task_module._resample_disk_store(task_module._resample_disk_path(("po", 0)), X, y, None)

    # the SPACE pre-evict phase runs BEFORE the temp is serialized
    assert "evict" in order and "savez" in order
    assert order.index("evict") < order.index("savez"), f"eviction did not precede write: {order}"


# ---------------------------------------------------------------------------
# 23. an impossible-to-fit artifact (bigger than the budget) must
#     NOT flush the cache — it skips without evicting, so existing entries survive.
# ---------------------------------------------------------------------------
def test_impossible_artifact_does_not_flush_cache(skewed_dataset, monkeypatch, capsys):
    # pre-populate two small entries under the default budget
    small = np.zeros((10, 2), dtype=np.float32)
    smally = np.zeros(10, dtype=np.int64)
    assert task_module._resample_disk_store(task_module._resample_disk_path(("keep", 0)), small, smally, None)
    assert task_module._resample_disk_store(task_module._resample_disk_path(("keep", 1)), small, smally, None)
    cache_dir = Path(task_module._resample_cache_dir())
    assert len(list(cache_dir.glob("*.npz"))) == 2

    # a big artifact whose RAW bytes already exceed a lowered budget
    big = np.zeros((100000, 4), dtype=np.float32)
    bigy = np.zeros(100000, dtype=np.int64)
    nbytes = big.nbytes + bigy.nbytes
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(nbytes // 2))  # budget < one artifact
    ev_before = task_module._resample_disk_evictions

    assert task_module._resample_disk_store(task_module._resample_disk_path(("big", 0)), big, bigy, None) is False
    assert "does not fit cache budget" in capsys.readouterr().out
    # the small entries SURVIVE and nothing was evicted
    assert len(list(cache_dir.glob("*.npz"))) == 2, "impossible artifact flushed the cache"
    assert task_module._resample_disk_evictions == ev_before


# ---------------------------------------------------------------------------
# 24. the byte budget is bounded by MEASURED free space
#     (min(configured, statvfs_free - reserve)) so the cache can't fill a
#     near-full filesystem and starve Ray spill / system writes.
# ---------------------------------------------------------------------------
def _fake_statvfs(free_bytes, frsize=4096):
    return SimpleNamespace(f_bavail=free_bytes // frsize, f_frsize=frsize)


def test_effective_budget_bounded_by_free_space(skewed_dataset, monkeypatch, capsys):
    cache_dir = task_module._resample_cache_dir()
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(32 * 1024 ** 3))
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES", str(1 * 1024 ** 3))
    monkeypatch.setattr(task_module.os, "statvfs", lambda p: _fake_statvfs(3 * 1024 ** 3))
    task_module._resample_fs_bound_warned_dirs.discard(cache_dir)

    eff = task_module._resample_effective_budget(cache_dir)
    assert eff == 2 * 1024 ** 3, "effective budget != free(3GiB) - reserve(1GiB)"
    out = capsys.readouterr().out
    assert "free-space bound is binding" in out
    assert "effective_budget=" in out


def test_effective_budget_configured_when_ample_space(skewed_dataset, monkeypatch):
    cache_dir = task_module._resample_cache_dir()
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(1 * 1024 ** 3))
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES", str(1 * 1024 ** 3))
    monkeypatch.setattr(task_module.os, "statvfs", lambda p: _fake_statvfs(100 * 1024 ** 3))
    # 100 GiB free - 1 GiB reserve = 99 GiB > 1 GiB configured -> configured governs
    assert task_module._resample_effective_budget(cache_dir) == 1 * 1024 ** 3


def test_store_skips_when_free_space_bound_binds(skewed_dataset, monkeypatch, capsys):
    cache_dir = Path(task_module._resample_cache_dir())
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(32 * 1024 ** 3))
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES", "0")
    monkeypatch.setattr(task_module.os, "statvfs", lambda p: _fake_statvfs(0))  # ~0 free
    task_module._resample_fs_bound_warned_dirs.discard(str(cache_dir))

    X = np.zeros((100, 4), dtype=np.float32)
    y = np.zeros(100, dtype=np.int64)
    ok = task_module._resample_disk_store(task_module._resample_disk_path(("fsb", 0)), X, y, None)
    assert ok is False, "store published despite the free-space bound"
    assert list(cache_dir.glob("*.npz")) == []
    assert "free-space bound is binding" in capsys.readouterr().out


def test_free_space_capacity_includes_existing_entries(skewed_dataset, monkeypatch):
    """statvfs free ALREADY excludes bytes held by cache entries, so
    max total = cache_bytes_now + free - reserve. The 14/8/6 example: 14 GiB free,
    8 GiB reserve, 6 GiB cached -> true capacity 12 GiB, so a 5 GiB artifact fits
    WITHOUT eviction (the buggy `free - reserve` = 6 GiB would have evicted)."""
    GiB = 1024 ** 3
    cache_dir = str(task_module._resample_cache_dir())
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(32 * GiB))
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES", str(8 * GiB))
    monkeypatch.setattr(task_module.os, "statvfs", lambda p: _fake_statvfs(14 * GiB))
    # 6 GiB already cached (fake entry — never actually evicted here)
    monkeypatch.setattr(
        task_module, "_resample_disk_entries",
        lambda d: [(os.path.join(cache_dir, "e.npz"), 6 * GiB, 1)],
    )
    monkeypatch.setattr(task_module, "_resample_temp_files", lambda d: [])
    task_module._resample_fs_bound_warned_dirs.discard(cache_dir)
    ev_before = task_module._resample_disk_evictions

    fits = task_module._evict_for_budget(cache_dir, 5 * GiB)
    assert fits is True, "5 GiB rejected despite 12 GiB true capacity (double-counted entries)"
    assert task_module._resample_disk_evictions == ev_before, "evicted usable entries within capacity"


def test_free_space_bound_still_binds_and_evicts(skewed_dataset, monkeypatch):
    """The corrected bound still binds when the cache is genuinely over it: with
    3 real entries of size s, free=s, reserve=2s -> capacity 2s, so a new s-sized
    artifact forces eviction of the two oldest down to the bound."""
    small = np.zeros((1000, 4), dtype=np.float32)
    sy = np.zeros(1000, dtype=np.int64)
    for i in range(3):
        assert task_module._resample_disk_store(task_module._resample_disk_path(("bind", i)), small, sy, None)
    cache_dir = Path(task_module._resample_cache_dir())
    s = list(cache_dir.glob("*.npz"))[0].stat().st_size

    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(32 * 1024 ** 3))
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_FS_RESERVE_BYTES", str(2 * s))
    monkeypatch.setattr(task_module.os, "statvfs", lambda p: _fake_statvfs(s, frsize=1))
    task_module._resample_fs_bound_warned_dirs.discard(str(cache_dir))
    ev_before = task_module._resample_disk_evictions

    fits = task_module._evict_for_budget(str(cache_dir), s)  # capacity 3s+s-2s = 2s
    assert fits is True
    assert task_module._resample_disk_evictions == ev_before + 2
    assert len(list(cache_dir.glob("*.npz"))) == 1


# ---------------------------------------------------------------------------
# 25. the prewarm releases the driver's in-process L1 RAM
#     cache after the durability pass — disk entries and provenance stay intact.
# ---------------------------------------------------------------------------
def test_prewarm_clears_driver_l1_keeps_disk_and_provenance(skewed_dataset, monkeypatch):
    import run_phase4_flower as runner
    monkeypatch.setattr(runner, "NUM_SUPERNODES", 3)

    _, summary = runner._prewarm_resample_cache(_RC3)

    assert len(task_module._resample_cache) == 0, "driver L1 RAM cache not released after prewarm"
    disk = list(Path(task_module._resample_cache_dir()).glob("*.npz"))
    assert len(disk) == 3, "disk entries were wrongly cleared (workers would recompute)"
    assert summary["persisted"] == 3, "provenance block corrupted by the L1 clear"
    assert summary["compute_only"] == 0


# ---------------------------------------------------------------------------
# 26. oldest-mtime eviction is LEAST-RECENTLY-USED — a disk
#     hit refreshes the entry's mtime, so a just-used current-run entry is not
#     evicted in favor of a newer FOREIGN-seed artifact in a shared/reused dir.
# ---------------------------------------------------------------------------
def _store_entry(key):
    X = np.zeros((1000, 4), dtype=np.float32)
    y = np.zeros(1000, dtype=np.int64)
    p = task_module._resample_disk_path(key)
    assert task_module._resample_disk_store(p, X, y, None)
    return Path(p)


def test_disk_hit_refreshes_recency_protects_current_set(skewed_dataset, monkeypatch):
    # two CURRENT-run entries, artificially given OLD mtimes
    cur = [_store_entry(("cur", i)) for i in range(2)]
    old = time.time_ns() - 100 * 1_000_000_000
    for p in cur:
        os.utime(p, ns=(old, old))
    # a NEWER foreign-seed artifact (fresh mtime)
    foreign = _store_entry(("foreign", 0))
    s = foreign.stat().st_size

    # Cross a filesystem timestamp tick before the loads: on fast hosts the
    # store and both disk loads can land inside ONE coarse-clock tick
    # (~4 ms observed on WSL2), giving the touched entries mtimes EQUAL to
    # the foreign write and turning the eviction order into a tiebreak.
    time.sleep(0.05)

    # USE the current entries (disk loads) -> their mtime is refreshed to now
    X = np.zeros((1000, 4), dtype=np.float32)
    y = np.zeros(1000, dtype=np.int64)
    for i in range(2):
        task_module._resample_cache.clear()  # force a DISK hit (not L1)
        _, _, _, was_cached = task_module._resample_cached(
            X, y, key=("cur", i), variant="random_under", target="balanced", seed=1)
        assert was_cached is True, "expected a disk hit on the current entry"

    # force eviction of ONE entry (capacity holds ~2 of the 3)
    cache_dir = Path(task_module._resample_cache_dir())
    monkeypatch.setenv("PRAXIS_RESAMPLE_CACHE_MAX_BYTES", str(int(2.5 * s)))
    task_module._evict_for_budget(str(cache_dir), 0)

    # the FOREIGN artifact (now the oldest-USED) is evicted; the current set lives
    assert not foreign.exists(), "recency ignored hits — foreign artifact survived"
    assert cur[0].exists() and cur[1].exists(), "just-used current entries were evicted"


def test_recency_touch_failure_degrades_quietly(skewed_dataset, monkeypatch):
    p = _store_entry(("touchfail", 0))
    assert p.exists()
    task_module._resample_cache.clear()

    def _boom(*_a, **_k):
        raise OSError(30, "read-only file system")

    monkeypatch.setattr(task_module.os, "utime", _boom)
    X = np.zeros((1000, 4), dtype=np.float32)
    y = np.zeros(1000, dtype=np.int64)
    # the disk hit still succeeds (touch failure is best-effort, never fatal)
    _, _, _, was_cached = task_module._resample_cached(
        X, y, key=("touchfail", 0), variant="random_under", target="balanced", seed=1)
    assert was_cached is True
    assert p.exists()
