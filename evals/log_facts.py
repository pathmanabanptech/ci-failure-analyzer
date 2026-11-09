"""Facts a raw GitHub Actions job log states about its own build.

Used to cross-check ``meta.yaml`` against the logs it points to (and by the capture
tool to fill ``base_sha`` and ``jdk``, which the runs API does not expose).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from guardrails.preprocess import ErrorKind, clean_log, find_termination

# actions/checkout prints the checked-out commit after running this command.
_GIT_LOG_CMD = re.compile(r"^\[command\]\S*git log -1 --format=%H\s*$")
_SHA_LINE = re.compile(r"^([0-9a-f]{40})$")
# pull_request builds check out GitHub's merge commit.
_MERGE = re.compile(r"^HEAD is now at [0-9a-f]+ Merge ([0-9a-f]{40}) into ([0-9a-f]{40})\b")
# actions/setup-java input echo: "  java-version: 17"
_JAVA_VERSION = re.compile(r"^\s+java-version:\s*['\"]?([\w.+-]+)['\"]?\s*$")


@dataclass(frozen=True)
class LogFacts:
    checked_out_sha: str | None  # push: head SHA; PR: GitHub's merge commit
    merge_head: str | None  # PR only
    merge_base: str | None  # PR only
    java_version: str | None
    termination: ErrorKind | None  # job cancelled / timed out


def extract_log_facts(raw: str) -> LogFacts:
    lines, _ = clean_log(raw)
    checked_out = merge_head = merge_base = java = None
    for i, line in enumerate(lines):
        text = line.text
        if checked_out is None and _GIT_LOG_CMD.match(text) and i + 1 < len(lines):
            m = _SHA_LINE.match(lines[i + 1].text)
            if m:
                checked_out = m.group(1)
        if merge_head is None and (m := _MERGE.match(text)):
            merge_head, merge_base = m.group(1), m.group(2)
        if java is None and (m := _JAVA_VERSION.match(text)):
            java = m.group(1)
    term = find_termination(lines)
    return LogFacts(
        checked_out_sha=checked_out,
        merge_head=merge_head,
        merge_base=merge_base,
        java_version=java,
        termination=term.kind if term else None,
    )
