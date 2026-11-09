"""G8 log pre-processing.

Turns a raw GitHub Actions job log into a compact excerpt for the model:

1. Clean each line: strip the runner timestamp prefix (and the ``job\\tstep\\t``
   prefix that ``gh run view --log`` adds), ANSI escape codes and ``##[group]`` /
   ``##[endgroup]`` markers.
2. Find the **first** fatal error (earliest line matching any fatal pattern).
3. Keep a window around that error plus the log's head and tail.

Every kept line carries its **original 1-based line number in the raw file**, so
evidence cited by the model can be verified against the raw log (G5).

Pure functions, no I/O. Redaction (G3) is a separate pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# --------------------------------------------------------------------- cleaning

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]")
_GH_CLI_PREFIX = re.compile(r"^[^\t\n]+\t[^\t\n]*\t(?=\d{4}-\d{2}-\d{2}T)")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")
_GROUP = re.compile(r"^##\[group\]")
_ENDGROUP = re.compile(r"^##\[endgroup\]\s*$")


def split_raw_lines(raw: str) -> list[str]:
    """Split like editors and ``grep -n`` do: on ``\\n`` only, so numbering matches."""
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def clean_line(line: str) -> str | None:
    """Return the cleaned line, or ``None`` if the line carries no content."""
    line = line.removeprefix("﻿")
    line = _GH_CLI_PREFIX.sub("", line, count=1)
    line = _TIMESTAMP.sub("", line, count=1)
    line = _ANSI.sub("", line)
    if _ENDGROUP.match(line):
        return None
    return _GROUP.sub("", line, count=1).rstrip()


# ---------------------------------------------------------------- fatal errors


class ErrorKind(StrEnum):
    javac_error = "javac_error"
    maven_error = "maven_error"
    gradle_failure = "gradle_failure"
    out_of_memory = "out_of_memory"
    cancelled = "cancelled"
    timeout = "timeout"


# Order matters only for ties on the same line (most specific first).
_FATAL_PATTERNS: tuple[tuple[ErrorKind, re.Pattern[str]], ...] = (
    (ErrorKind.out_of_memory, re.compile(r"\bOutOfMemoryError\b")),
    (ErrorKind.timeout, re.compile(r"exceeded the maximum execution time", re.I)),
    (ErrorKind.timeout, re.compile(r"^##\[error\].*\btimed? ?out\b", re.I)),
    (ErrorKind.cancelled, re.compile(r"The operation was cancell?ed", re.I)),
    (ErrorKind.javac_error, re.compile(r"\.(?:java|kt|groovy):\d+: error: ")),
    (ErrorKind.javac_error, re.compile(r"^error: ")),
    (ErrorKind.maven_error, re.compile(r"^\[ERROR\] \S")),
    (ErrorKind.gradle_failure, re.compile(r"^FAILURE: ")),
    (ErrorKind.gradle_failure, re.compile(r"^\* What went wrong:")),
)

_GH_ERROR = re.compile(r"^##\[error\]")


@dataclass(frozen=True)
class LogLine:
    number: int  # 1-based line number in the raw log
    text: str  # cleaned text


@dataclass(frozen=True)
class FatalError:
    kind: ErrorKind
    line: LogLine


def find_first_fatal(lines: list[LogLine]) -> FatalError | None:
    for line in lines:
        for kind, pattern in _FATAL_PATTERNS:
            if pattern.search(line.text):
                return FatalError(kind=kind, line=line)
    return None


_TERMINATION_KINDS = frozenset({ErrorKind.cancelled, ErrorKind.timeout})


def find_termination(lines: list[LogLine]) -> FatalError | None:
    """The job-level cancel/timeout marker (last one), if the job was cut short.

    Reported separately from the first fatal error: a job can fail first (e.g. an
    OOM in a test) and then hang until it is cancelled or times out.
    """
    for line in reversed(lines):
        for kind, pattern in _FATAL_PATTERNS:
            if kind in _TERMINATION_KINDS and pattern.search(line.text):
                return FatalError(kind=kind, line=line)
    return None


def find_all_fatal(lines: list[LogLine]) -> list[FatalError]:
    """Every line matching a fatal pattern (first matching kind per line)."""
    found: list[FatalError] = []
    for line in lines:
        for kind, pattern in _FATAL_PATTERNS:
            if pattern.search(line.text):
                found.append(FatalError(kind=kind, line=line))
                break
    return found


# --------------------------------------------------------------------- excerpt


@dataclass(frozen=True)
class ExcerptConfig:
    head: int = 25
    before: int = 40
    after: int = 80
    tail: int = 40


@dataclass(frozen=True)
class Excerpt:
    lines: tuple[LogLine, ...]
    first_error: FatalError | None
    termination: FatalError | None  # job cancelled / timed out
    total_lines: int  # raw line count

    def render(self) -> str:
        """Line-numbered text; gaps are marked with how many raw lines were skipped.

        ``>`` marks the first fatal error, ``!`` the cancel/timeout marker.
        """
        width = len(str(self.total_lines))
        first = self.first_error.line.number if self.first_error else None
        term = self.termination.line.number if self.termination else None
        out: list[str] = []
        prev = 0
        for line in self.lines:
            if line.number > prev + 1:
                out.append(f"{'':>{width}}  ... [{line.number - prev - 1} lines omitted] ...")
            marker = ">" if line.number == first else "!" if line.number == term else "|"
            out.append(f"{line.number:>{width}}{marker} {line.text}")
            prev = line.number
        if prev < self.total_lines:
            out.append(f"{'':>{width}}  ... [{self.total_lines - prev} lines omitted] ...")
        return "\n".join(out)


def clean_log(raw: str) -> tuple[list[LogLine], int]:
    """Cleaned content lines with original numbers, and the raw line count."""
    raw_lines = split_raw_lines(raw)
    cleaned: list[LogLine] = []
    for number, text in enumerate(raw_lines, start=1):
        c = clean_line(text)
        if c is not None:
            cleaned.append(LogLine(number=number, text=c))
    return cleaned, len(raw_lines)


def preprocess(raw: str, config: ExcerptConfig | None = None) -> Excerpt:
    cfg = config or ExcerptConfig()
    lines, total = clean_log(raw)
    first = find_first_fatal(lines)
    termination = find_termination(lines)

    keep: set[int] = set(range(min(cfg.head, len(lines))))

    # Nothing after the last ##[error] line (e.g. "Process completed with exit code 1")
    # is kept: it is post-job cleanup noise that would crowd out the build summary.
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if _GH_ERROR.match(lines[i].text):
            end = i + 1
            break
    keep.update(range(max(0, end - cfg.tail), end))

    if first is not None:
        idx = lines.index(first.line)
        keep.update(range(max(0, idx - cfg.before), min(end, idx + cfg.after + 1)))
        keep.add(idx)  # always, even if it sits after the last ##[error]
    if termination is not None:
        keep.add(lines.index(termination.line))

    return Excerpt(
        lines=tuple(lines[i] for i in sorted(keep)),
        first_error=first,
        termination=termination,
        total_lines=total,
    )
