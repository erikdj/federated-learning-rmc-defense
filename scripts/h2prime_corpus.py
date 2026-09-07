"""Corpus layer for the H2′ confirmatory read — custody in, rows out.

The assembly map, the sealed-seed authority, the § 2.2a rotation, the per-file
loader and the design matrix. Everything that decides WHICH rows are scored
lives here; nothing here decides what the scores mean.

Split out of `adjudicate_h2prime.py` (2026-08-12) for file size; behaviour
unchanged, verified by a byte-identical dev-smoke output across the split.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from h2prime_common import (
    ATTACKS, CONFIRMATORY, DEV_SEEDS, FEATS, H2_CONFIRM_SEED_KEY,
    H2_CONFIRM_SEED_PATH, H2_CONFIRM_SEED_SHA256, HardStop, Profile, R, Refusal,
    REGISTERED_DEFENSE_TOKEN, SCENARIOS, SEALED_SEED_PATH,
    SEALED_SEED_SHA256, canonical_device_id,
)

# --------------------------------------------------------------------------
# gate 1 — the assembly map
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    scenario: str          # long form, e.g. s0_clean_baseline
    scen: str              # short form, e.g. S0
    seed: int
    source: str            # EXP-051 / EXP-053
    path: str
    stem: str
    defense: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sealed_seeds() -> list[int]:
    """The sealed seed set, in ascending numeric order (§ 2.2a rotation order)."""
    if not SEALED_SEED_PATH.is_file():
        raise Refusal(f"sealed seed authority missing: {SEALED_SEED_PATH}")
    actual = _sha256_file(SEALED_SEED_PATH)
    if actual != SEALED_SEED_SHA256:
        raise Refusal(
            "sealed seed authority sha256 mismatch (EXP-051 § 2.1):\n"
            f"  recorded {SEALED_SEED_SHA256}\n  actual   {actual}"
        )
    seeds = json.loads(SEALED_SEED_PATH.read_text())["seeds"]
    return sorted(int(s) for s in seeds)


def h2_confirm_seeds() -> list[int]:
    """EXP-048's REGISTERED seed universe, ascending — the § 4 secondary 9 grid.

    Unsealed (EXP-048 was read and published 2026-08-09), but registered: the
    expected staging grid is derived from this committed file, never from the
    staged directory the gate is validating.
    """
    if not H2_CONFIRM_SEED_PATH.is_file():
        raise Refusal(f"EXP-048 seed authority missing: {H2_CONFIRM_SEED_PATH}")
    actual = _sha256_file(H2_CONFIRM_SEED_PATH)
    if actual != H2_CONFIRM_SEED_SHA256:
        raise Refusal(
            "EXP-048 seed authority sha256 mismatch:\n"
            f"  recorded {H2_CONFIRM_SEED_SHA256}\n  actual   {actual}"
        )
    doc = json.loads(H2_CONFIRM_SEED_PATH.read_text())
    return sorted(int(s) for s in doc[H2_CONFIRM_SEED_KEY])


def load_assembly_map(map_path: Path, profile: Profile) -> tuple[list[Cell], dict]:
    """Parse and validate the custody assembly map. Refuses loudly, never guesses."""
    if not map_path.is_file():
        raise Refusal(f"assembly map not found: {map_path}")
    try:
        doc = json.loads(map_path.read_text())
    except json.JSONDecodeError as exc:
        raise Refusal(f"assembly map is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or "cells" not in doc:
        raise Refusal("assembly map has no 'cells' array")

    raw = doc["cells"]
    if len(raw) != profile.n_cells:
        raise Refusal(
            f"assembly map has {len(raw)} cells; {profile.name} requires exactly "
            f"{profile.n_cells}"
        )

    cells: list[Cell] = []
    seen: set[tuple[str, int]] = set()
    undigested: list[str] = []
    for entry in raw:
        for key in ("scenario", "seed", "path"):
            if key not in entry:
                raise Refusal(f"assembly-map cell missing '{key}': {entry}")
        scenario = str(entry["scenario"]).lower()
        seed = int(entry["seed"])
        if scenario not in R.SCEN_SHORT:
            raise Refusal(f"unknown scenario in assembly map: {scenario}")
        key = (scenario, seed)
        if key in seen:
            raise Refusal(f"duplicate (scenario, seed) cell in assembly map: {key}")
        seen.add(key)

        path = Path(entry["path"])
        if not path.is_file():
            raise Refusal(f"assembly-map file missing on disk: {path}")
        if path.stat().st_size == 0:
            raise Refusal(f"assembly-map file is empty: {path}")

        stem = path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
        parts = stem.split("__")
        if len(parts) != 4:
            raise Refusal(f"unit-id stem is not scenario__defense__exec__seed: {stem}")
        f_scen, defense, _exec, f_seedtok = parts
        if f_scen.lower() != scenario:
            raise Refusal(f"filename scenario {f_scen} != map scenario {scenario}: {path}")
        if f_seedtok != f"seed{seed}":
            raise Refusal(f"filename seed {f_seedtok} != map seed {seed}: {path}")

        recorded = entry.get("sha256")
        # `digest_unavailable` is an EXPLICIT declaration that the bytes were
        # never fetched (a dry-run map). It is treated exactly like an absent
        # digest — a stated absence is still an absence, and the whole point of
        # the marker is that such a map must not reach the sealed read.
        if entry.get("digest_unavailable"):
            recorded = None
        if not recorded:
            # Collected, not raised per-entry: a map missing digests is
            # usually missing ALL of them, and failing on the first says
            # nothing about the scale of the problem.
            undigested.append(f"{scenario}x{seed} ({path.name})")
        else:
            actual = _sha256_file(path)
            if actual != recorded:
                raise Refusal(
                    f"assembly-map sha256 mismatch for {path}:\n"
                    f"  recorded {recorded}\n  actual   {actual}"
                )

        cells.append(
            Cell(
                scenario=scenario,
                scen=R.SCEN_SHORT[scenario],
                seed=seed,
                source=str(entry.get("source", "unknown")),
                path=str(path),
                stem=stem,
                defense=defense,
            )
        )

    # CONTENT DIGESTS ARE MANDATORY ON THE ADJUDICATING PROFILE. Path
    # resolution proves a file is THERE, not that it is the registered one —
    # and the sealed corpus is opened exactly once, so a substituted or
    # re-generated file cannot be caught after the fact. The old conditional
    # verified a digest when present and shrugged when absent, which meant a
    # hand-written or dry-run-produced map silently downgraded custody to
    # paths. See Profile.requires_content_digest for the DEV-SMOKE exemption
    # and its reason.
    if undigested and profile.requires_content_digest:
        raise Refusal(
            f"assembly-map entries carry NO sha256 content digest on the "
            f"{profile.name} profile ({len(undigested)} of {len(raw)}): "
            + ", ".join(undigested[:8])
            + (" ..." if len(undigested) > 8 else "")
            + ". The sealed corpus is read exactly once, so file identity has "
            "to be pinned by CONTENT before the read — a path only proves a "
            "file exists. Re-run scripts/build_exp051_assembly_map.py, which "
            "always emits digests, and re-run the dry-run."
        )

    defenses = sorted({c.defense for c in cells})
    if len(defenses) != 1:
        raise Refusal(
            f"heterogeneous defense configuration across the corpus: {defenses} — "
            "a mixed-arm corpus is a custody failure (EXP-051 § 5 item 5)"
        )
    # HOMOGENEITY AND IDENTITY. A uniformly wrong token passes the check above
    # and would adjudicate the sealed cohort on an unregistered configuration —
    # permanently, since the corpus is scored exactly once.
    if defenses[0] != REGISTERED_DEFENSE_TOKEN:
        raise Refusal(
            f"defense token {defenses[0]!r} is not the REGISTERED configuration "
            f"{REGISTERED_DEFENSE_TOKEN!r} (EXP-051 § 5 item 5). The corpus is "
            "homogeneous but on the wrong arm; nothing is adjudicated."
        )

    scens = sorted({c.scen for c in cells})
    if scens != SCENARIOS:
        raise Refusal(f"scenario set {scens} != the frozen five {SCENARIOS}")

    seeds = sorted({c.seed for c in cells})
    if len(seeds) != profile.n_seeds:
        raise Refusal(
            f"{len(seeds)} distinct seeds; {profile.name} requires exactly "
            f"{profile.n_seeds}"
        )
    if len(cells) != len(SCENARIOS) * len(seeds):
        raise Refusal("assembly map is not the exact scenario × seed cross product")

    sealed = sealed_seeds()
    if profile is CONFIRMATORY:
        if seeds != sealed:
            raise Refusal(
                "assembly-map seeds are not the sealed confirmatory set "
                "(data/h2prime_confirm_seeds.json is the authority)"
            )
    else:
        if seeds != sorted(DEV_SEEDS):
            raise Refusal(f"DEV-SMOKE requires the five dev seeds {sorted(DEV_SEEDS)}")
        overlap = sorted(set(seeds) & set(sealed))
        if overlap:
            raise Refusal(f"DEV-SMOKE refuses to touch sealed seeds: {overlap}")

    meta = {
        "map_path": str(map_path),
        "map_sha256": _sha256_file(map_path),
        "map_meta": doc.get("_meta", {}),
        "defense_token": defenses[0],
        "seeds_ascending": seeds,
        "cells_by_source": {
            s: sum(1 for c in cells if c.source == s)
            for s in sorted({c.source for c in cells})
        },
    }
    return cells, meta


# --------------------------------------------------------------------------
# gate 2 — the § 2.2a rotation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Rotation:
    i: int                 # 1-based rotation index
    test: int
    calibration: int
    fit: tuple[int, int, int]


def rotation_plan(seeds_ascending: list[int]) -> list[Rotation]:
    """§ 2.2a frozen index formula, 1-based, defined for any n ≥ 5.

    test = s_i ; calibration = s_{1+(i mod n)} ;
    fit  = s_{1+((i+1) mod n)}, s_{1+((i+2) mod n)}, s_{1+((i+3) mod n)}
    """
    n = len(seeds_ascending)
    if n < 5:
        raise Refusal(f"rotation requires n ≥ 5 seeds; got {n}")

    def s(idx_1based: int) -> int:
        return seeds_ascending[idx_1based - 1]

    plan = []
    for i in range(1, n + 1):
        test = s(i)
        calib = s(1 + (i % n))
        fit = tuple(s(1 + ((i + k) % n)) for k in (1, 2, 3))
        chosen = {test, calib, *fit}
        if len(chosen) != 5:
            raise Refusal(f"rotation {i} does not select five distinct seeds: {chosen}")
        plan.append(Rotation(i=i, test=test, calibration=calib, fit=fit))
    scored = sorted(r.test for r in plan)
    if scored != seeds_ascending:
        raise Refusal("rotation does not score every seed exactly once")
    return plan


# --------------------------------------------------------------------------
# loading — assembly map in, rows out (mirrors revalidate_v115.load())
# --------------------------------------------------------------------------
def load_cells(cells: list[Cell]) -> list[dict]:
    """One `derive_window_feats()` invocation per (scenario, seed, defense) file.

    Mirrors `revalidate_v115.load()` exactly — the same filename↔row identity
    assertions and the same per-file window derivation — except that files are
    selected by the custody assembly map rather than by glob (EXP-053 § 2.3: a
    prefix sync would silently overwrite the two refill cells). Files are
    consumed in ascending unit-id order, which is the order `sorted(glob(...))`
    would produce for a single directory.
    """
    rows: list[dict] = []
    for cell in sorted(cells, key=lambda c: c.stem):
        with open(cell.path, "r", encoding="utf-8") as fh:
            frows = [json.loads(line) for line in fh if line.strip()]
        if not frows:
            raise Refusal(f"signal log has zero rows: {cell.path}")
        for r in frows:
            if r["seed"] != cell.seed:
                raise Refusal(f"row seed {r['seed']} != filename seed {cell.seed}: {cell.path}")
            if R.SCEN_SHORT[r["scenario"].lower()] != cell.scen:
                raise Refusal(f"row scenario {r['scenario']} != filename scenario: {cell.path}")
            r["_scen"] = cell.scen
            r["_seed"] = cell.seed
            r["_defense"] = cell.defense
            r["_source"] = cell.source
        R.derive_window_feats(frows)
        rows.extend(frows)
    return rows


def design_matrix(rs: list[dict]) -> np.ndarray:
    """§ 2.1a: X = the frozen 9-column order, float64, no preprocessing.

    § 2.1a item 5 — a null in a RAW feature is a HARD STOP, not a handled case.
    """
    X = np.array([[r[f] for f in FEATS] for r in rs], dtype=float)
    if X.size and not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        first = rs[int(bad[0][0])]
        raise HardStop(
            "§ 2.1a item 5 HARD STOP: non-finite value in the design matrix. "
            f"{len(bad)} offending entries; first at feature "
            f"'{FEATS[int(bad[0][1])]}' of row "
            f"(scenario={first['_scen']}, seed={first['_seed']}, "
            f"logical_cid={first.get('logical_cid')}, "
            f"scenario_round={first.get('scenario_round')}). Scoring stops; the "
            "condition is reported. No imputation or drop rule is pre-registered."
        )
    return X



def exposed_devices(rows: list[dict]) -> dict[str, set[str]]:
    """v1.15b § 3 — per family, the canonical device lineages that carry it.

    "Every row of any device whose lineage carries the held-out family anywhere
    in the corpus, under any alias": the key is `canonical_device_id`, so
    `client_5` and `client_5_new2` are ONE identity and excluding the lineage
    excludes every alias's rows.
    """
    out: dict[str, set[str]] = {A: set() for A in ATTACKS}
    for r in rows:
        A = r.get("attack_type")
        if r["malicious_gt"] and A in out:
            out[A].add(canonical_device_id(r["logical_cid"]))
    return out

