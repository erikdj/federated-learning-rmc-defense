"""Shared dataclasses for scripts/audit_calibration_gate.py.

Kept in its own module (repo convention: many small files, see
scripts/audit_run_instrumentation.py / scripts/analyze_ramp_selection.py for the
sibling-module-import style this package follows).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnitRef:
    """A calibration-gate view of one matrix unit.

    Deliberately NOT praxis_exp.units.Unit: that dataclass has no notion of
    "expected optimizer_state" (the runner's internal short-form vocabulary,
    see _calibration_gate_lib.OPTIMIZER_STATE_TOKEN) or defense_token (the
    signal-log filename token from docker/entrypoint.py's _DEFENSE_TOKEN).
    S3 mode builds this from a manifest Unit; local-dirs mode builds it from
    a downloaded result JSON's own "config"/"seed" fields (there is no
    manifest to source them from offline).
    """

    unit_id: str
    config: str
    scenario: str
    seed: int
    expected_optimizer_state: str  # "persistent" | "reset" -- runner's short form
    defense_token: str  # e.g. "krum", "trustscore", "krumtge", "tgensemble"
    # Declared FL rounds for this unit (manifest Unit.rounds in S3 mode; the
    # --rounds CLI value in local-dirs mode). The result trajectory must cover
    # server rounds 0..rounds+1 -- see _calibration_gate_lib.check_rounds_consistency.
    rounds: int


@dataclass
class CheckResult:
    """One pass/fail line item in the gate report.

    `required=False` marks checks that are reported but never fail the gate
    exit code (Szeląg gate, wall-clock sizing) -- spec calls these
    "exit-code-neutral, human review gates".
    """

    name: str
    unit_id: str | None
    passed: bool
    required: bool
    detail: str


@dataclass
class GateReport:
    exp_id: str
    mode: str  # "s3" | "local"
    results: list[CheckResult] = field(default_factory=list)
    wall_clock_rows: list[dict] | None = None
    fanout_projection: dict | None = None
    szelag: dict | None = None

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def extend(self, results: list[CheckResult]) -> None:
        self.results.extend(results)

    @property
    def required_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.required and not r.passed]

    @property
    def ok(self) -> bool:
        """Exit-code predicate: True iff every REQUIRED check passed.

        Szeląg gate and wall-clock sizing are intentionally excluded (they are
        stored on separate GateReport fields, not as CheckResults, precisely so
        they can never contribute to required_failures).
        """
        return len(self.required_failures) == 0
