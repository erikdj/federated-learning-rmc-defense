"""Markdown report rendering for scripts/audit_calibration_gate.py.

One render function produces the text used for BOTH stdout and the markdown
report file, so the two outputs can never drift from each other.
"""
from __future__ import annotations

from _calibration_gate_types import GateReport

_SZELAG_BANNER = "=" * 72


def render_report(report: GateReport) -> str:
    lines: list[str] = []
    lines.append(f"# Calibration gate report -- {report.exp_id} ({report.mode} mode)")
    lines.append("")

    required = [r for r in report.results if r.required]
    passed = [r for r in required if r.passed]
    failed = [r for r in required if not r.passed]
    skipped = [r for r in report.results if not r.required]

    lines.append(f"**Required checks: {len(passed)}/{len(required)} passed.**")
    if failed:
        lines.append("")
        lines.append("## FAILED (required)")
        for r in failed:
            scope = f"[{r.unit_id}] " if r.unit_id else ""
            lines.append(f"- **{r.name}** {scope}-- {r.detail}")
    lines.append("")
    lines.append("## All checks")
    lines.append("")
    lines.append("| check | unit | required | passed | detail |")
    lines.append("|---|---|---|---|---|")
    for r in report.results:
        lines.append(
            f"| {r.name} | {r.unit_id or '-'} | {r.required} | "
            f"{'PASS' if r.passed else 'FAIL'} | {r.detail} |"
        )

    if report.szelag is not None:
        s = report.szelag
        verdict = "PASS" if s["passed"] else "FAIL"
        lines.append("")
        lines.append("## Szeląg gate (informational -- does not affect exit code)")
        lines.append("")
        lines.append(_SZELAG_BANNER)
        lines.append(f"SZELĄG GATE: {verdict} -- {s['detail']}")
        lines.append(_SZELAG_BANNER)

    if report.wall_clock_rows is not None:
        lines.append("")
        lines.append("## Wall-clock sizing (ESTIMATE -- does not affect exit code)")
        lines.append("")
        lines.append("| unit_id | config | elapsed_seconds |")
        lines.append("|---|---|---|")
        for row in report.wall_clock_rows:
            lines.append(f"| {row.get('unit_id', '-')} | {row.get('config', '-')} | {row.get('elapsed_seconds', '-')} |")
        proj = report.fanout_projection or {}
        if proj:
            lines.append("")
            lines.append(f"_{proj.get('note', 'ESTIMATE')}_")
            if "projected_wall_clock_hours" in proj:
                lines.append("")
                lines.append(
                    f"- observed units: {proj['n_units_observed']}, "
                    f"mean elapsed: {proj['mean_elapsed_seconds']:.1f}s\n"
                    f"- fan-out target: {proj['fanout_target_units']} units at "
                    f"{proj['concurrency']} concurrent jobs (32 vCPUs / 8 vCPUs per job)\n"
                    f"- projected wall-clock: {proj['projected_wall_clock_seconds']:.0f}s "
                    f"(~{proj['projected_wall_clock_hours']:.2f}h) over {proj['waves']} wave(s)"
                )

    lines.append("")
    lines.append(f"**Gate exit status: {'PASS' if report.ok else 'FAIL'}** "
                  f"({len(failed)} required failure(s); {len(skipped)} non-required check(s) reported above)")
    return "\n".join(lines)
