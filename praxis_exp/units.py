"""Canonical matrix-unit expansion and deterministic unit identifiers."""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Unit", "unit_id", "expand_matrix"]

_SAFE_TOKEN = re.compile(r"^[a-z0-9_]+$")


def _token(value: str) -> str:
    token = value.lower().replace("+", "_").replace(" ", "")
    if not _SAFE_TOKEN.fullmatch(token):
        raise ValueError(
            f"unit token {token!r} (from {value!r}) has unexpected characters; expected [a-z0-9_]"
        )
    return token


def unit_id(config: str, scenario: str, mode: str, seed: int, repeat: int = 0) -> str:
    """Deterministic, S3/filesystem-safe identifier for one matrix unit.

    ``repeat`` is a sentinel-guarded suffix for the same-seed replicate axis
    repeat >= 1 appends ``__rep{repeat}``; repeat == 0 (the
    default, and the only value emitted when the repeats axis is inactive) is
    byte-for-byte the pre-axis id. This IFF is load-bearing — the id string is
    the chain-of-custody key for idempotent-skip markers, MLflow tags, and S3
    artifact paths, so it must not drift for any existing (non-replicated) matrix.
    """
    r = int(repeat)
    if r < 0:
        # Fail fast at the identity boundary: a negative repeat would fall
        # through to the base id and silently alias onto the un-suffixed unit
        # (marker / MLflow tag / S3 key collision with a real non-replicate).
        raise ValueError(f"unit_id: repeat must be >= 0 (0 = no replicate axis), got {repeat}")
    base = f"{_token(scenario)}__{_token(config)}__{_token(mode)}__seed{int(seed)}"
    return f"{base}__rep{r}" if r >= 1 else base


@dataclass(frozen=True)
class Unit:
    config: str
    scenario: str
    mode: str
    seed: int
    max_per_client: int
    rounds: int
    array_index: int
    # LAST field, default 0, so existing positional constructors and manifests
    # written without a `repeat` key (Unit(**u)) still parse. 0 == replicate
    # axis inactive (no __rep suffix); >=1 is the 1-based replicate ordinal.
    repeat: int = 0

    @property
    def unit_id(self) -> str:
        return unit_id(self.config, self.scenario, self.mode, self.seed, self.repeat)


def expand_matrix(
    configs: list[str],
    scenarios: list[str],
    seeds: list[int],
    mode: str,
    max_per_client: int,
    rounds: int,
    repeats: int = 1,
) -> list[Unit]:
    """Expand a (config x scenario x seed x repeat) matrix into ordered Units.

    Iteration order is configs -> scenarios -> seeds -> repeat; array_index is
    the position in that stable order and is how AWS_BATCH_JOB_ARRAY_INDEX maps
    back to a unit. ``repeats`` (default 1) is the same-seed replicate count:
    when repeats == 1 the replicate axis is inactive and each Unit carries
    repeat == 0 (no __rep suffix) — byte-identical to the pre-axis expansion.
    When repeats > 1, each (config,scenario,seed) yields repeats units with
    1-based repeat ordinals rep1..rep{repeats}.
    """
    if not configs or not scenarios or not seeds:
        raise ValueError("expand_matrix: configs, scenarios, and seeds must all be non-empty")
    if repeats < 1:
        raise ValueError(f"expand_matrix: repeats must be >= 1, got {repeats}")
    units: list[Unit] = []
    index = 0
    for config in configs:
        for scenario in scenarios:
            for seed in seeds:
                for r in range(1, repeats + 1):
                    repeat = r if repeats > 1 else 0
                    units.append(
                        Unit(config, scenario, mode, int(seed), max_per_client, rounds, index, repeat)
                    )
                    index += 1
    return units
