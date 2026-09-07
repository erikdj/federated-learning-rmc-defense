#!/usr/bin/env python3
"""Schema-v5 signal logs → the H3 primary scorer's event corpus (DRYRUN F4).

The adapter between the two ends of the H3 measurement chain. The left-hand end
is `flowerfl/signal_logger.py`, which emits one JSONL row per (client, round)
and — on a re-entry event row only — populates the ten-field re-entry contract
of amendment v1.10 § 5.1. The right-hand end is
`scripts/analyze_h3_relink.py`, the FROZEN primary scorer, which consumes one
row per re-entry EVENT. Until this script existed nothing bridged them: the
dry-run found `reentry_event_key` in four files, none of which was an extractor.

WHAT THIS SCRIPT IS RESPONSIBLE FOR
-----------------------------------
1. **Refusing v4 logs by name.** Dry-run § 1.2/1.3: v4 logs structurally cannot
   feed this pipeline. The fingerprint was never on the wire and all twelve v5
   fields — `aggregation_coefficient` most consequentially — are absent. A v4
   log is not a degraded input to be tolerated; it is the wrong instrument.
2. **Never dropping an event silently.** A dropped event shrinks a denominator,
   and a silently shrunk denominator BIASES the metric rather than merely
   degrading it — the one failure mode that cannot be detected downstream. Every
   malformed, half-populated or unknown-provenance event row is a hard refusal
   naming the row; only rows that are *not* events at all are skipped, and their
   count is reported.
3. **Naming the arm field `h3_arm`, never `arm`.** `arm` is already taken in
   this codebase for the resampling manifest's variant label
   (`flowerfl/resampling_manifest.py:35,66`, `flowerfl/client_app.py:248`;
   values like `smote@balanced` / `off`), and that label rides in the same
   signal-log row as the defense token. Two meanings under one key is how a
   control-arm row silently ends up in an adjudicating denominator. The corpus
   therefore carries the defense token as `h3_arm`, the resampling label — when
   present — is passed through under the unambiguous name `resampling_arm`, and
   a bare `arm` key never reaches the corpus at all. The scorer refuses a row
   that carries `arm` without `h3_arm` rather than reading one as the other.
4. **Deduplicating on `reentry_event_key`**, exactly as the scorer does, so the
   two ends agree on what "one event" means.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
* **It does not apply the arm rule.** Every arm's events are extracted; the
  scorer applies "TGE+FP only" (v1.10 § 5.1). Filtering here as well would put
  the rule in two places, and a corpus that silently contained only one arm
  could not be audited for the presence of the control arm.
* **It does not map or normalise the defense token.** `h3_arm` is the row's
  `defense` value verbatim. Which token denotes the +FP arm (`tgefp`,
  `krumtgefp`, …) is a scoring-time selection made with the scorer's `--arm`
  flag, not a translation invented here.
* **It reads nothing but the log.** No fingerprints, no τ, no sealed material,
  no enforcement outcome. The corpus is the logged registry assertions, which is
  what makes the primary metric detector-independent (§ 5.1).

INPUT CONTRACT
--------------
One or more schema-v5 signal-log JSONL files. Every row must carry
`signal_log_schema_version == 5`. A row is a RE-ENTRY EVENT iff its
`reentry_event_key` is non-null; ordinary participation rows carry the ten
contract fields as explicit nulls and are skipped.

OUTPUT CONTRACT
---------------
JSONL, one row per deduplicated re-entry event, carrying exactly
``EVENT_CORPUS_FIELDS``: the scorer's eleven `REQUIRED_FIELDS`, plus `scenario`
and `h3_arm` (which the scorer's scenario tally and arm rule read), plus
`seed`, `run_uid`, `source_log` and `resampling_arm` for provenance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from flowerfl.signal_logger import (  # noqa: E402
    REENTRY_EVENT_FIELDS,
    SCHEMA_V5_ADDED_FIELDS,
    SIGNAL_LOG_SCHEMA_VERSION,
)
from scripts.analyze_h3_relink import (  # noqa: E402
    H3_ARM_FIELD,
    REQUIRED_FIELDS,
    RESAMPLING_ARM_FIELD,
    UNUSABLE_VALUES,
)

# ===========================================================================
# Contract constants
# ===========================================================================

#: The only signal-log schema this extractor accepts. Bound to the logger's own
#: constant so the two cannot drift apart unnoticed.
SUPPORTED_SCHEMA_VERSION: int = SIGNAL_LOG_SCHEMA_VERSION

#: The v5 field whose absence identifies a v4 log in the refusal message — the
#: load-bearing GWU-9 quantity the H3 rejoin rule is defined on.
V5_MARKER_FIELD = "aggregation_coefficient"

#: The row-level field carrying the config's defense token, which becomes the
#: corpus's `h3_arm`.
DEFENSE_FIELD = "defense"

#: ADDITIVE and OPTIONAL: the nearest candidate the registry considered,
#: recorded on every re-entry decision including one that asserted no match
#: (`signal_logger.REENTRY_NEAREST_FIELDS`).
#:
#: Emitted ONLY when the source log carries the key. Absent-not-null is the
#: whole contract: an ABSENT field means "this corpus predates the extension and
#: cannot answer threshold-free rank-1", while an explicit NULL means "the
#: candidate pool was empty", which is a real and determinate observation.
#: Collapsing the two would let a pre-extension corpus read as though every
#: event had had no candidate.
OPTIONAL_NEAREST_FIELDS: Tuple[str, ...] = (
    "nearest_entry_id",
    "nearest_logical_id",
)

#: Emitted for every event, in this order. The optional pair is appended per-row
#: when present and is deliberately NOT in `REQUIRED_FIELDS` — the frozen D7
#: contract does not move.
EVENT_CORPUS_FIELDS: Tuple[str, ...] = tuple(REQUIRED_FIELDS) + (
    "scenario",
    H3_ARM_FIELD,
    "seed",
    "run_uid",
    "source_log",
    "resampling_arm",
) + OPTIONAL_NEAREST_FIELDS

#: Fields compared when two rows share a `reentry_event_key`. Provenance fields
#: are excluded: the same event legitimately appears in a re-read of the same
#: log, and `source_log` differing is not a disagreement about the EVENT.
_CONFLICT_FIELDS: Tuple[str, ...] = tuple(REQUIRED_FIELDS) + ("scenario", H3_ARM_FIELD)

#: Event fields that must be non-null on every event row.
#: * the truth half and the registry DECISION, because the scorer classifies on
#:   them and a null would silently misclassify;
#: * `tau`, because a null τ means the registry ran without a locked threshold,
#:   which integrity gate (c) forbids;
#: * `generation`, which is an int on every entry (0 for a new one).
#: `min_d` is NOT here: an overflowed or absent distance reads as no-match and
#: `_jsonify` writes non-finite floats as null, so a null `min_d` is truthful on
#: an unmatched event. It is required non-null when a match IS asserted.
_REQUIRED_NON_NULL = (
    "reentry_event_key",
    "current_cid",
    "gt_logical_id",
    "gt_is_malicious",
    "asserted_match",
    "tau",
    "generation",
)

#: Non-null only when `asserted_match` is true.
_REQUIRED_WHEN_MATCHED = (
    "asserted_parent_entry_id",
    "asserted_parent_logical_id",
    "min_d",
)

#: Run-level provenance that must be usable on every event row (gate (b)).
_REQUIRED_RUN_FIELDS = ("scenario", DEFENSE_FIELD)

#: Fields whose populated presence proves a row IS an event even when the key
#: is null — the null-`run_uid` case `ScenarioStrategy._reentry_event_fields`
#: warns about, which would otherwise cost the corpus an event.
_EVENT_EVIDENCE_FIELDS = ("current_cid", "gt_logical_id", "gt_is_malicious")


class ExtractionError(RuntimeError):
    """Raised when a signal log cannot be turned into a scorable event corpus."""


# ===========================================================================
# Reading
# ===========================================================================

def _iter_rows(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    """Yield `(line_number, row)` for a JSONL signal log."""
    if not path.exists():
        raise ExtractionError(f"signal log not found: {path}")
    text = path.read_text()
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ExtractionError(
                f"{path}:{line_number} is not a JSON object — a signal log is "
                "one JSON object per line"
            )
        yield line_number, row


def _check_schema_version(path: Path, line_number: int, row: Mapping[str, Any]) -> None:
    """Version-GATE the row. v4 is refused by name, never migrated."""
    if "signal_log_schema_version" not in row:
        raise ExtractionError(
            f"{path}:{line_number} carries no 'signal_log_schema_version' — "
            f"refusing to guess. This extractor reads schema v"
            f"{SUPPORTED_SCHEMA_VERSION} logs only"
        )
    version = row["signal_log_schema_version"]
    if version == SUPPORTED_SCHEMA_VERSION:
        return
    raise ExtractionError(
        f"{path}:{line_number} is a schema v{version} signal log; this "
        f"extractor reads schema v{SUPPORTED_SCHEMA_VERSION} only. A schema v4 "
        "log structurally CANNOT feed the H3 pipeline: the fingerprint was "
        "never on the wire, and all "
        f"{len(SCHEMA_V5_ADDED_FIELDS)} v5 fields — including "
        f"'{V5_MARKER_FIELD}', the quantity the H3 rejoin rule is defined on — "
        "are absent. v1.10 § 5.1: no H3 result may be computed from "
        "'effective_weight=num_examples' masquerading as a coefficient. Re-run "
        "the scenario on a v5 build; there is no reader change that fixes this"
    )


# ===========================================================================
# Event recognition and validation
# ===========================================================================

def _is_event_row(row: Mapping[str, Any]) -> bool:
    """True iff this row is a re-entry event (or a corrupt attempt at one).

    A participation row carries all ten contract fields as explicit nulls. A row
    with ANY of the truth half populated is an event, even if its key is null —
    that case is a refusal below, never a skip, because skipping it would shrink
    a denominator without a trace.
    """
    if row.get("reentry_event_key") is not None:
        return True
    return any(row.get(field) is not None for field in _EVENT_EVIDENCE_FIELDS)


def _validate_event(path: Path, line_number: int, row: Mapping[str, Any]) -> None:
    where = f"{path}:{line_number}"

    missing = [f for f in REENTRY_EVENT_FIELDS if f not in row]
    if missing:
        raise ExtractionError(
            f"{where} is a re-entry event row but is missing schema-v5 "
            f"field(s) {missing} — the v5 contract emits every field on every "
            "row, so a MISSING key is a defect (an explicit null is not)"
        )
    if "server_round" not in row:
        raise ExtractionError(
            f"{where} is a re-entry event row with no 'server_round' — the "
            "eleventh field of the v1.10 § 5.1 event contract"
        )

    for field in _REQUIRED_NON_NULL:
        if row.get(field) is None:
            if field == "tau":
                # The OBSERVE-ONLY PRE-LOCK posture writes exactly this shape
                # (`min_d = inf`, no τ, so `_jsonify` nulls both). It is a
                # legitimate LOG shape and an illegitimate SCORING shape: every
                # event would be an unmatched miss, recall would be 0, and H3
                # would falsify by arithmetic with zero information content.
                # Refuse loudly here rather than hand that corpus to the scorer.
                raise ExtractionError(
                    f"{where} is a re-entry event row with a null 'tau' — the "
                    "observe-only PRE-LOCK shape (no locked threshold in "
                    "force). Integrity gate (c) requires a locked τ, and a "
                    "corpus of pre-lock events scores recall 0 by "
                    "construction, falsifying H3 by arithmetic. Extract only "
                    "runs made under a locked τ"
                )
            raise ExtractionError(
                f"{where} is a re-entry event row whose {field!r} is null. "
                "Refusing to emit it: a scorer that dedups on "
                "'reentry_event_key' and classifies on the truth half would "
                "drop or misclassify this event, silently shrinking a "
                "denominator — the one failure mode that biases the metric"
            )

    if bool(row["asserted_match"]):
        for field in _REQUIRED_WHEN_MATCHED:
            if row.get(field) is None:
                raise ExtractionError(
                    f"{where} asserts a match but its {field!r} is null — the "
                    "row is not scorable (v1.10 § 5.1: the registry half is "
                    "written for every event, regardless of any downstream "
                    "enforcement action)"
                )

    for field in _REQUIRED_RUN_FIELDS:
        value = row.get(field)
        if value is None or str(value) in UNUSABLE_VALUES:
            raise ExtractionError(
                f"{where} has UNKNOWN/missing run provenance in {field!r} — "
                "integrity gate (b) requires zero such rows"
            )

    for field in ("reentry_event_key", "current_cid", "gt_logical_id"):
        if str(row[field]) in UNUSABLE_VALUES:
            raise ExtractionError(
                f"{where} has UNKNOWN/missing provenance in {field!r} — "
                "integrity gate (b) requires zero such rows"
            )


def _to_corpus_row(path: Path, row: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a validated v5 event row onto the corpus contract.

    Built from an explicit field list rather than by copying the row, so no
    signal-log key — the resampling manifest's `arm` above all — can reach the
    corpus under a name the scorer might read as something else.
    """
    corpus: Dict[str, Any] = {field: row.get(field) for field in REQUIRED_FIELDS}
    corpus["scenario"] = row["scenario"]
    # The H3 arm is the DEFENSE token. It is never taken from `arm`, which in a
    # signal-log row is the resampling manifest's variant label.
    corpus[H3_ARM_FIELD] = str(row[DEFENSE_FIELD])
    corpus["seed"] = row.get("seed")
    corpus["run_uid"] = row.get("run_uid")
    corpus["source_log"] = str(path)
    corpus["resampling_arm"] = row.get(RESAMPLING_ARM_FIELD)
    # Additive optional pair: copied only when the source row HAS the key, so a
    # pre-extension log yields a corpus without it rather than one full of nulls
    # that would read as "no candidate existed".
    for field in OPTIONAL_NEAREST_FIELDS:
        if field in row:
            corpus[field] = row[field]
    return corpus


# ===========================================================================
# Extraction
# ===========================================================================

def extract_events(
    paths: Sequence[Path],
    census: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Read schema-v5 signal logs and return the deduplicated event corpus.

    Args:
        paths: schema-v5 signal-log JSONL files.
        census: optional dict, filled in place with the row/event accounting so
            a caller can report exactly what was read, skipped and emitted —
            nothing is dropped without appearing in this tally.

    Raises:
        ExtractionError: on any wrong-schema, malformed, half-populated,
            unknown-provenance or self-contradicting row.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    rows_read = 0
    non_event_rows = 0
    duplicate_rows = 0

    for raw_path in paths:
        path = Path(raw_path)
        for line_number, row in _iter_rows(path):
            rows_read += 1
            _check_schema_version(path, line_number, row)
            if not _is_event_row(row):
                non_event_rows += 1
                continue
            _validate_event(path, line_number, row)

            corpus_row = _to_corpus_row(path, row)
            key = str(corpus_row["reentry_event_key"])
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = corpus_row
                continue
            conflicting = [
                f for f in _CONFLICT_FIELDS if existing.get(f) != corpus_row.get(f)
            ]
            if conflicting:
                raise ExtractionError(
                    f"{path}:{line_number} conflict under reentry_event_key "
                    f"{key!r}: rows disagree on {conflicting} (first seen in "
                    f"{existing['source_log']}). One event key must describe "
                    "exactly one event"
                )
            duplicate_rows += 1

    events = list(by_key.values())
    if census is not None:
        census.update(
            {
                "logs": [str(Path(p)) for p in paths],
                "rows_read": rows_read,
                "non_event_rows_skipped": non_event_rows,
                "duplicate_event_rows_collapsed": duplicate_rows,
                "events_emitted": len(events),
                "events_by_arm": dict(Counter(e[H3_ARM_FIELD] for e in events)),
                "events_by_scenario": dict(Counter(e["scenario"] for e in events)),
                "malicious_events": sum(1 for e in events if bool(e["gt_is_malicious"])),
                "honest_events": sum(1 for e in events if not bool(e["gt_is_malicious"])),
            }
        )
    return events


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the H3 re-entry event corpus from schema-v5 signal logs, "
            "for scripts/analyze_h3_relink.py. Refuses v4 logs: the fingerprint "
            "was never on the wire and the v5 fields do not exist there."
        )
    )
    parser.add_argument(
        "--logs", nargs="+", required=True, type=Path,
        help="schema-v5 signal-log JSONL files (one run per file)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="event-corpus JSONL to write (default: stdout)",
    )
    parser.add_argument(
        "--census-out", type=Path, default=None,
        help="optional JSON file recording what was read, skipped and emitted",
    )
    args = parser.parse_args(argv)

    census: Dict[str, Any] = {}
    try:
        events = extract_events(args.logs, census=census)
    except ExtractionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    payload = "".join(json.dumps(event) + "\n" for event in events)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload)
    else:
        sys.stdout.write(payload)

    print(
        f"rows read: {census['rows_read']}  "
        f"non-event rows skipped: {census['non_event_rows_skipped']}  "
        f"duplicate event rows collapsed: "
        f"{census['duplicate_event_rows_collapsed']}  "
        f"events: {census['events_emitted']} "
        f"(malicious {census['malicious_events']}, "
        f"honest {census['honest_events']})"
    )
    print(f"  by arm ({H3_ARM_FIELD}): {census['events_by_arm']}")
    print(f"  by scenario: {census['events_by_scenario']}")
    if args.out:
        print(f"written: {args.out}")
    if args.census_out:
        Path(args.census_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.census_out).write_text(json.dumps(census, indent=2) + "\n")
        print(f"census: {args.census_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
