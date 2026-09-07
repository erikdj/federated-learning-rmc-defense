"""Regression gate that would have caught the v8 cache failure.

The v8 process-local LRU resample cache was refuted at FLEET SHAPE, not by any
unit test — because the failure lives in Ray's real ``ActorPool`` LIFO
scheduling (no client->actor affinity), which a mocked pool hides entirely.
This test stands up a REAL ``ray.util.actor_pool.ActorPool`` at the production
shape property (actors > concurrent clients) and drives the ACTUAL production
cache code path (``flowerfl.task._resample_cached``) across several fit/evaluate
rounds.

Measured facts it pins (see tests/test_resample_cache_fleet_shape.py):
  * v8 (process-local LRU only): ~16% hit rate, no warm-up -> this test's
    ``hit_rate(round>=2) >= 0.9`` assertion FAILS.
  * v9 (node-local disk cache shared by all actor processes): once ANY actor
    computes a client's resample in round 1, EVERY actor finds it on disk in
    later rounds regardless of which actor the scheduler hands it to -> ~100%
    hit rate from round 2.
  * per-process memory does not scale with actor count: each actor's in-RAM L1
    stays bounded by ``_RESAMPLE_CACHE_MAXSIZE``; the disk layer is the
    equalizer, so adding actors never multiplies resident array copies.

If someone reverts ``_resample_cached`` to a process-local-only cache, the
cross-actor round-2 hit rate collapses and this test goes red.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ray = pytest.importorskip("ray", reason="ray required for fleet-shape probe")
from ray.util.actor_pool import ActorPool  # noqa: E402

import flowerfl.task as task_module  # noqa: E402

# Production shape PROPERTY: more actors than concurrent clients (fleet is
# 32 actors / 21 clients). Downscaled hard for CI — 4 actors /
# 3 clients / 3 rounds — but the ACTORS > CLIENTS inequality (what defeats a
# per-process LRU) is preserved, so the regression protection is intact.
#
# THIS TEST MUST RUN BEFORE ANY RESAMPLE-CACHE-DESIGN CHANGE. In this repo the
# local full suite is the merge gate, and an excluded test is a dead test, so it
# is deliberately NOT marked `slow`: keeping it in the default gate is the whole
# point. (If the real-Ray wall-clock ever regresses past ~20s in a future env,
# marking it `slow` is the fallback — but then it must be run manually on any
# cache change.)
NUM_ACTORS = 4
NUM_CLIENTS = 3
NUM_ROUNDS = 3


@ray.remote
class ResampleProbeActor:
    """Drives the real production cache path with tiny synthetic arrays."""

    def __init__(self, cache_dir: str):
        # Every actor process shares ONE node-local disk cache dir (the v9
        # design); each keeps its own process-local L1 (mirrors production —
        # a Ray actor is its own OS process).
        os.environ["PRAXIS_RESAMPLE_CACHE_DIR"] = cache_dir
        import numpy as np
        # A small imbalanced 2-class split: 15 majority / 5 minority (ratio 0.33
        # < 1) so random_under actually resamples on a miss and writes disk.
        rng = np.random.default_rng(0)
        self._X = rng.normal(size=(20, 4)).astype(np.float32)
        self._y = np.array([0] * 15 + [1] * 5, dtype=np.int64)
        self._seen = set()  # DISTINCT partitions this actor has constructed

    def run_client(self, client_id: int, round_idx: int, stage: str):
        import flowerfl.task as task
        self._seen.add(int(client_id))
        key = ("fleet_probe", int(client_id))
        _, _, _, was_cached = task._resample_cached(
            self._X, self._y, key=key,
            variant="random_under", target="balanced", seed=1234 + client_id,
        )
        # bytes RETAINED by this actor's L1 right now (the memory-ceiling metric)
        l1_bytes = sum(
            int(getattr(a, "nbytes", 0))
            for entry in task._resample_cache.values()
            for a in entry[:2]  # (X_res, y_res)
        )
        return {
            "pid": os.getpid(),
            "client_id": client_id,
            "round": round_idx,
            "stage": stage,
            "hit": bool(was_cached),
            "l1_size": len(task._resample_cache),
            "l1_bytes": l1_bytes,
            "distinct_seen": len(self._seen),
        }


def _run_stage(pool, round_idx, stage):
    for cid in range(NUM_CLIENTS):
        pool.submit(
            lambda a, v: a.run_client.remote(v[0], v[1], v[2]),
            (cid, round_idx, stage),
        )
    return [pool.get_next_unordered() for _ in range(NUM_CLIENTS)]


@pytest.mark.ray
def test_disk_cache_wins_at_fleet_shape(tmp_path):
    assert NUM_ACTORS > NUM_CLIENTS, "shape property: actors must exceed clients"
    cache_dir = tmp_path / "fleet_cache"
    cache_dir.mkdir()
    task_module._reset_resample_cache()

    ray.init(
        num_cpus=NUM_ACTORS,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {
            "PYTHONPATH": str(PROJECT_ROOT),
            "PRAXIS_RESAMPLE_CACHE_DIR": str(cache_dir),
        }},
        ignore_reinit_error=True,
    )
    try:
        actors = [ResampleProbeActor.remote(str(cache_dir)) for _ in range(NUM_ACTORS)]
        pool = ActorPool(actors)
        results = []
        for r in range(NUM_ROUNDS):
            results += _run_stage(pool, r, "fit")
            results += _run_stage(pool, r, "eval")
    finally:
        ray.shutdown()

    # --- hit rate from round 2 onward (post warm-up) ---
    warm = [x for x in results if x["round"] >= 2]
    warm_hits = sum(x["hit"] for x in warm)
    hit_rate = warm_hits / len(warm)

    # per-(round,stage) for the failure message
    by_rs = defaultdict(lambda: [0, 0])
    for x in results:
        by_rs[(x["round"], x["stage"])][0] += x["hit"]
        by_rs[(x["round"], x["stage"])][1] += 1
    detail = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_rs.items())}

    assert hit_rate >= 0.9, (
        f"v9 disk cache must hit >=90% from round 2 at fleet shape "
        f"(actors={NUM_ACTORS} > clients={NUM_CLIENTS}); got {hit_rate:.3f}. "
        f"A process-local-only cache collapses here. per-(round,stage): {detail}"
    )

    # --- distinct actors actually used (proves genuine unpinned scheduling) ---
    distinct_pids = {x["pid"] for x in results}
    assert len(distinct_pids) > 1, "probe did not exercise multiple actors"

    # --- MEMORY CEILING: retained L1 bytes do NOT scale with distinct
    # partitions seen (the promise from the original brief, made real). At fleet
    # shape a cap-4 L1 x 32 persistent actors x ~1.18 GB would be ~151 GB > the
    # 120 GiB container (the v8 H-C failure); cap=1 pins it to ~38 GB. Here: at
    # least one actor constructs MORE distinct partitions than the cap, yet its
    # L1 length and retained bytes stay bounded by cap.
    cap = task_module._RESAMPLE_CACHE_MAXSIZE

    from flowerfl.smote_resampler import resample_training_split
    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(20, 4)).astype(np.float32)
    _y = np.array([0] * 15 + [1] * 5, dtype=np.int64)
    _Xr, _yr, _ = resample_training_split(_X, _y, variant="random_under", target="balanced", seed=1234)
    artifact_bytes = int(_Xr.nbytes) + int(_yr.nbytes)
    assert artifact_bytes > 0

    max_distinct = max(x["distinct_seen"] for x in results)
    assert max_distinct > cap, (
        f"test not meaningful: no actor constructed more than cap={cap} distinct "
        f"partitions (max seen {max_distinct}) — cannot prove non-scaling"
    )

    max_l1 = max(x["l1_size"] for x in results)
    assert max_l1 <= cap, (
        f"per-actor L1 grew to {max_l1} entries > cap {cap}; RAM scales with load"
    )

    max_l1_bytes = max(x["l1_bytes"] for x in results)
    assert max_l1_bytes <= cap * artifact_bytes, (
        f"per-actor retained L1 bytes {max_l1_bytes} exceed cap({cap}) x "
        f"artifact({artifact_bytes}) = {cap * artifact_bytes}; memory scales with "
        f"distinct-partitions-seen (max {max_distinct})"
    )
