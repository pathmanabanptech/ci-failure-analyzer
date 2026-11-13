"""Notifier that prints the report (local runs and dry runs).

Mirrors the PR notifier's dedupe semantics (G10): a second report with the same
``sha:category`` key in one process is printed as an update, not a new alert.
"""

from __future__ import annotations

import sys
from typing import TextIO

from adapters.base import Report


class ConsoleNotifier:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._seen: set[str] = set()

    def notify(self, report: Report) -> None:
        d = report.diagnosis
        update = report.dedupe_key in self._seen
        self._seen.add(report.dedupe_key)

        lines = [
            f"=== CI failure analysis{' (updated)' if update else ''}: "
            f"{d.status.value} / {d.category.value} (confidence {d.confidence:.2f}) ===",
            f"repo: {report.repo}  sha: {report.head_sha[:12]}  "
            f"builds: {', '.join(report.build_tools) or '-'}  key: {report.dedupe_key}",
            f"Root cause: {d.root_cause}",
            f"Why it passed locally: {d.local_vs_ci}",
            "Evidence:",
        ]
        for ev in d.evidence or ():
            where = f"{ev.source}:{ev.line}" if ev.line is not None else ev.source
            mark = "verified" if ev.verified else "unverified"
            lines.append(f"  - {where} ({mark}): {ev.snippet}")
        if not d.evidence:
            lines.append("  (none)")
        if d.fix_diff:
            lines += ["Suggested fix:", "```diff", d.fix_diff.rstrip("\n"), "```"]
        if d.copilot_fixable:
            lines.append("Copilot-fixable: yes")
        print("\n".join(lines), file=self._stream)
