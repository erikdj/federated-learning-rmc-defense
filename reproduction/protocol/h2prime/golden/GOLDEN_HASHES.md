# GOLDEN_HASHES — frozen window-feature fixture (spec v1.15 rev-6 § 2.2b)

Pre-registered golden-input / golden-output gate for the H2' confirmatory.
Passing `tests/test_window_feats_golden.py` is a PRE-CONDITION of the
confirmatory scoring pass; a hash mismatch is a HARD STOP per § 2.2b.

- **Frozen construction:** `derive_window_feats()` in
  `.planning/h2prime/protocol_exact_revalidation/revalidate_v115.py`,
  frozen at commit `dcef0f7` ("evidence(h2prime): commit the protocol-exact
  revalidation evidence base (rev-4 grounding)"). Verified byte-identical to
  that commit at generation time (`git diff dcef0f7 -- …/revalidate_v115.py`
  empty).
- **Generated at repo commit:** `0bd8b52` (working tree of `dcef0f7`-frozen
  builder; generation date 2026-08-10).
- **Golden input provenance:** 79 verbatim raw rows (byte-exact
  lines, nothing stripped) from the EXP-011 dev signal log
  `s3_identity_reset_only__krum_tge__persistent_optimizer__seed42.jsonl`
  (dev seed 42; sealed/confirmatory data untouched). Clients kept in original
  file order: full histories of `client_2` (malicious, rounds 1–7),
  `client_2_new1` (RMC rejoin identity, rounds 9–17), `client_2_new4`
  (rejoin, rounds 39–50), `client_9` (honest, rounds 1–50), plus exactly one
  row of `client_17` (round 1) as the ≤1-observation minimum-period case.
- **Structural coverage:** multi-client; window saturation (>3 rounds, both
  honest and malicious); ≤1-observation 0.0 convention (single-row client and
  every episode's first round); the RMC rejoin lineage `client_2` →
  `client_2_new1` → `client_2_new4` (fresh `logical_cid` per rejoin — the
  harness's actual RMC manifestation, so each rejoin is a fresh episode via
  the per-`logical_cid` grouping key).
- **Noted absences (none present anywhere in the 100-file EXP-011 dev
  corpus, verified by exhaustive scan 2026-08-10):** no equal-`scenario_round`
  tie within a `logical_cid` (the frozen tie-break `(scenario_round,
  original_file_row_index)` is therefore pinned by the spec text but not
  exercisable with real rows); no within-cid tenure reset (rejoins mint fresh
  `logical_cid`s, so the builder's `tenure ≤ prev` episode-clear branch is
  defensive and unexercised); no per-feature nulls in `update_norm`,
  `train_loss`, or `cos_to_median`. The fixture is raw rows verbatim per the
  build rule — no synthetic rows were fabricated to force these branches.
- **Canonical serialization:** the expected file IS the canonical bytes —
  `json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)`
  UTF-8, rows in the frozen total order (per-`logical_cid` groups in
  first-appearance order, sorted ascending by `scenario_round`, ties by
  original row index). Hashing the file ≡ hashing the canonical
  serialization.

```
window_feats_golden_input.jsonl  sha256 = b955b3eca522b1361a7bc39732cf8609c86968886369dddee18b5591506562f3
window_feats_golden_expected.json  sha256 = 82a55a94b912b41d59ec842ebd5c94b85bd348a995f8db14e8f7dd58c2ae0eb8
```
