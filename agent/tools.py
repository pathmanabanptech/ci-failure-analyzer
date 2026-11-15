"""Read-only agent tools. Each wraps an adapter; none touches the environment directly.

Every result that carries build content (log lines, file contents, commit subjects)
is redacted and wrapped in an ``<untrusted_...>`` block. Closing-tag look-alikes
inside the content are defused so it cannot end its block early.

Tools are built per diagnosis by ``build_tools`` and bound to that run's SHAs: the
model picks ``ref="head"``, ``"base"`` or ``"merge"`` (the merge of head into base that
a pull_request build compiles; head for push builds), never an arbitrary revision.
"""

# No ``from __future__ import annotations``: Strands reads the type hints at runtime.

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from strands import tool

from adapters.base import AdapterError, RepoReader, RunDetails
from agent.base import run_attempts_text
from guardrails.preprocess import LogLine
from guardrails.redact import redact
from guardrails.verify_evidence import COMMIT_PREFIX

Ref = Literal["head", "base", "merge"]
CommitRef = Literal["head", "base"]

MAX_LOG_LINES = 150
MAX_FILE_LINES = 300
MAX_COMMITS = 20

_TAG_LOOKALIKE = re.compile(r"<(/?)(untrusted)", re.I)


def defuse(text: str) -> str:
    """Make ``<untrusted...`` / ``</untrusted...`` inert inside untrusted content."""
    return _TAG_LOOKALIKE.sub(r"<\\\1\2", text)


def untrusted(kind: str, body: str, **attrs: str | int) -> str:
    attr_text = "".join(f' {k}="{defuse(str(v))}"' for k, v in attrs.items())
    return f"<untrusted_{kind}{attr_text}>\n{defuse(body)}\n</untrusted_{kind}>"


def number_lines(lines: Sequence[tuple[int, str]]) -> str:
    width = len(str(lines[-1][0])) if lines else 1
    return "\n".join(f"{n:>{width}}| {text}" for n, text in lines)


@dataclass
class ToolContext:
    run: RunDetails
    log_lines: tuple[LogLine, ...]
    repo: RepoReader | None
    # Context blocks shown to the model, for the evidence verifier.
    shown: dict[str, list[str]] = field(default_factory=dict)
    # Per-commit text shown to the model (full sha -> lines), for kind="commit" evidence.
    commits: dict[str, list[str]] = field(default_factory=dict)

    def show_commit(self, sha: str, lines: list[str]) -> None:
        self.commits.setdefault(sha, []).extend(lines)
        self.shown.setdefault("commit", []).extend(lines)

    def sha(self, ref: str) -> str | None:
        if ref == "head":
            return self.run.head_sha
        if ref == "base":
            return self.run.base_sha
        return None

    def read(self, ref: str, path: str) -> tuple[str, str]:
        """(content, note) of ``path`` at ``ref``. Raises AdapterError / ValueError."""
        if self.repo is None:
            raise ValueError("repository not available")
        run = self.run
        if ref == "merge":
            if run.base_sha is None:  # push build: CI compiled the head commit itself
                return self.repo.read_file(run.head_sha, path), "push build: merge = head"
            return self.repo.read_file_at_merge(run.base_sha, run.head_sha, path), ""
        sha = self.sha(ref)
        if sha is None:
            raise ValueError(f"this run has no {ref} commit")
        return self.repo.read_file(sha, path), ""

    def context(self) -> dict[str, str]:
        ctx = {k: "\n".join(v) for k, v in self.shown.items()}
        ctx.update({f"{COMMIT_PREFIX}{sha}": "\n".join(v) for sha, v in self.commits.items()})
        return ctx


def build_tools(ctx: ToolContext) -> list[Any]:
    @tool
    def read_log_excerpt(start_line: int, end_line: int) -> str:
        """Read log lines start_line..end_line (inclusive, original log line numbers).

        Use only for lines outside the excerpt you were given. At most 150 lines.

        Args:
            start_line: First log line number to read.
            end_line: Last log line number to read.
        """
        if end_line < start_line:
            return "error: end_line must be >= start_line"
        end_line = min(end_line, start_line + MAX_LOG_LINES - 1)
        lines = [(x.number, x.text) for x in ctx.log_lines if start_line <= x.number <= end_line]
        if not lines:
            return f"error: no log lines in {start_line}-{end_line}"
        return untrusted("log", number_lines(lines), lines=f"{start_line}-{end_line}")

    @tool
    def read_file_at_sha(
        path: str, ref: Ref = "head", start_line: int = 1, end_line: int = 0
    ) -> str:
        """Read a repository file at the failing commit, with line numbers.

        Allowed: build files (pom.xml, build.gradle, settings.gradle, gradle.properties),
        src/**, .github/workflows/**, .mvn/**, gradle/**. At most 300 lines per call.

        Args:
            path: Repo-relative path, e.g. src/main/java/com/example/Foo.java or pom.xml.
            ref: "head" = the PR/pushed commit; "base" = the PR base; "merge" = head merged
                into base, which is what a pull_request build actually compiled.
            start_line: First line to return (1-based).
            end_line: Last line to return; 0 means up to 300 lines from start_line.
        """
        try:
            content, ref_note = ctx.read(ref, path)
        except (AdapterError, ValueError) as e:
            return f"error: {e}"
        content = redact(content)
        all_lines = content.split("\n")
        start = max(1, start_line)
        stop = min(len(all_lines), end_line if end_line > 0 else start + MAX_FILE_LINES - 1)
        stop = min(stop, start + MAX_FILE_LINES - 1)
        lines = [(n, all_lines[n - 1]) for n in range(start, stop + 1)]
        if not lines:
            return f"error: {path} has {len(all_lines)} lines"
        note = f"lines {start}-{stop} of {len(all_lines)}"
        attrs: dict[str, str | int] = {"path": path, "ref": ref, "lines": note}
        if ref_note:
            attrs["note"] = ref_note
        return untrusted("file", number_lines(lines), **attrs)

    @tool
    def recent_commits(ref: CommitRef = "head", limit: int = 10, path: str = "") -> str:
        """List recent commits reachable from the failing commit (newest first).

        Args:
            ref: "head" = the commit that failed; "base" = the PR base.
            limit: Number of commits (max 20).
            path: Only commits touching this repo-relative path; empty for all.
        """
        if ctx.repo is None:
            return "error: repository not available"
        sha = ctx.sha(ref)
        if sha is None:
            return f"error: this run has no {ref} commit"
        try:
            commits = ctx.repo.recent_commits(
                sha, limit=max(1, min(limit, MAX_COMMITS)), path=path or None
            )
        except AdapterError as e:
            return f"error: {e}"
        lines = []
        for c in commits:
            line = f"{c.sha[:12]} {c.authored_at.date().isoformat()} {redact(c.subject)}"
            ctx.show_commit(c.sha, [line])
            lines.append(line)
        return untrusted("commits", "\n".join(lines) or "(none)", ref=ref)

    @tool
    def symbol_history(identifier: str, path: str = "") -> str:
        """Find commits that changed calls or declarations of a method or constructor.

        Searches diffs for lines matching `identifier(` (git log -G). For pull_request
        builds it covers commits reachable from head and commits reachable from base
        since the two diverged; for push builds, recent history of head. Each result
        shows the commit, which side it is on ([head] or [base]), and the added (+) and
        removed (-) lines that mention the identifier. Use it when the error names a
        method that exists on head but not in the merge, e.g. renamed on base.

        Args:
            identifier: A plain method/constructor name, e.g. getPet (not getPet(String,boolean)).
            path: Only this repo-relative file; empty for all files.
        """
        if ctx.repo is None:
            return "error: repository not available"
        run = ctx.run
        try:
            changes = ctx.repo.symbol_history(run.base_sha, run.head_sha, identifier, path or None)
        except AdapterError as e:
            return f"error: {e}"
        out: list[str] = []
        for c in changes:
            block = [f"{c.sha[:12]} [{c.side}] {redact(c.subject)}"]
            block += [f"  - {x.path}: {redact(x.text)}" for x in c.removed]
            block += [f"  + {x.path}: {redact(x.text)}" for x in c.added]
            ctx.show_commit(c.sha, block)
            out.extend(block)
        scope = "base...head" if run.base_sha else "head history (push build)"
        return untrusted(
            "symbol_history", "\n".join(out) or "(no commits)", identifier=identifier, scope=scope
        )

    @tool
    def run_attempts() -> str:
        """Attempts of this run and other runs of the same commit, with their conclusions."""
        return run_attempts_text(ctx.run)

    return [read_log_excerpt, read_file_at_sha, recent_commits, symbol_history, run_attempts]
