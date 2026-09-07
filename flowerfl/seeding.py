"""Deterministic seeding utilities.

Before this module existed, the experiment `seed` from run_config was recorded
in filenames and signal logs but was **never wired into any RNG** — torch's
global generator (and numpy/random) stayed seeded from OS entropy, so the
initial global model built in `server_app.server_fn` and every client's
per-round training draws differed process-to-process. Runs were therefore
only *statistically* reproducible, never byte-identical.

Two small helpers fix that:

- `seed_everything(seed)` seeds the three RNGs that feed every numeric path
  here (`random`, `numpy`, `torch`). Call it on the server before model
  construction, and on each client at the top of `fit`.
- `derive_seed(base, *components)` composes a stable per-(client, round)
  seed so distinct clients/rounds get distinct-but-reproducible streams.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's `random`, NumPy, and torch global RNGs from `seed`.

    These three cover every stochastic numeric path in this codebase: model
    weight initialisation (torch), dropout masks (torch), DataLoader shuffle
    (torch global RNG when no explicit generator is passed), and any
    numpy/random-based attack or scheduling draw.

    Note on PYTHONHASHSEED: we deliberately do NOT set it here. Once the
    interpreter is running, mutating `os.environ["PYTHONHASHSEED"]` is a no-op
    for the current process (hash randomisation is fixed at interpreter
    start-up), and nothing in the numeric pipeline depends on `hash` salting
    anyway — the per-(client, round) composition below uses SHA-256, not the
    salted built-in `hash`.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))  # numpy's legacy seed must fit in uint32
    torch.manual_seed(seed)


def derive_seed(base: int, *components: int) -> int:
    """Deterministically compose a child seed from `base` and `components`.

    Used to give each (client, round) pair its own reproducible RNG stream,
    e.g. `derive_seed(base_seed, partition_id, server_round)`.

    The composition is a SHA-256 over the tuple repr reduced mod 2**31, so it
    is:
      - stable across processes and platforms (unlike the built-in `hash`,
        which is salted per-process via PYTHONHASHSEED),
      - order-sensitive (client/round order matters),
      - bounded to a non-negative value accepted by `torch.manual_seed` and
        `random.seed`.
    """
    payload = repr((int(base), *(int(c) for c in components))).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest, 16) % (2**31)
