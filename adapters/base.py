"""The four I/O interfaces. Agent logic never touches the environment directly.

Phase 1 ships local implementations; Phase 3 adds GitHub ones and Phase 4 AWS ones
behind the same Protocols, so agent code, prompts and evals do not change.

Content returned by ``LogSource`` and ``RepoReader`` (logs, file contents, commit
messages) is untrusted. Adapters return it raw; callers redact secrets and wrap it
in delimited untrusted blocks before it reaches a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from agent.schema import BuildTool, Category, Diagnosis

Conclusion = Literal["success", "failure", "cancelled", "timed_out"]
Event = Literal["push", "pull_request", "workflow_dispatch"]


# ------------------------------------------------------------------------ errors


class AdapterError(Exception):
    """Base class for adapter failures the agent may see as a tool error."""


class NotFoundError(AdapterError):
    """Run, log, commit or file does not exist."""


class AccessDeniedError(AdapterError):
    """Request rejected by policy (path allowlist, traversal, invalid ref)."""


# -------------------------------------------------------------------- LogSource


@dataclass(frozen=True)
class RunRef:
    """Identifies one CI job run. ``job_id`` picks the job when a run has several."""

    repo: str  # owner/name
    run_id: int
    job_id: int | None = None


@dataclass(frozen=True)
class RunAttempt:
    attempt: int
    conclusion: Conclusion | None  # None = not recorded (only the latest is known)


@dataclass(frozen=True)
class RelatedRun:
    """Another run of the same head SHA (e.g. the branch build that passed)."""

    build_tool: BuildTool
    run_id: int
    event: Event
    conclusion: Conclusion
    base_sha: str | None = None


@dataclass(frozen=True)
class RunDetails:
    ref: RunRef
    build_tool: BuildTool
    event: Event
    head_sha: str
    base_sha: str | None  # pull_request runs: the base the merge was built against
    conclusion: Conclusion
    runner_os: str
    attempts: tuple[RunAttempt, ...]
    related_runs: tuple[RelatedRun, ...] = ()


@runtime_checkable
class LogSource(Protocol):
    def get_log(self, run: RunRef) -> str:
        """The job's raw log, exactly as the CI system returns it."""
        ...

    def get_run(self, run: RunRef) -> RunDetails:
        """Run metadata: SHAs, event, conclusion, attempts, related runs."""
        ...


# ------------------------------------------------------------------- RepoReader


@dataclass(frozen=True)
class Commit:
    sha: str
    authored_at: datetime
    subject: str  # untrusted; no author name/email (not needed, PII)


@dataclass(frozen=True)
class SymbolLine:
    path: str
    text: str  # untrusted


@dataclass(frozen=True)
class SymbolChange:
    """A commit whose diff adds or removes a line mentioning an identifier."""

    sha: str
    subject: str  # untrusted
    side: Literal["head", "base"]  # head: in the PR, not in base; base: on base since the fork
    added: tuple[SymbolLine, ...]  # "+" lines containing the identifier
    removed: tuple[SymbolLine, ...]  # "-" lines containing the identifier


@dataclass(frozen=True)
class PatchCheck:
    ok: bool
    message: str  # "applies" or why not (git's message)


@runtime_checkable
class RepoReader(Protocol):
    def read_file(self, sha: str, path: str) -> str:
        """File content at a commit. Raises AccessDeniedError / NotFoundError."""
        ...

    def read_file_at_merge(self, base_sha: str, head_sha: str, path: str) -> str:
        """File content in the merge of ``head_sha`` into ``base_sha``: what a
        pull_request build compiles. Raises AccessDeniedError / NotFoundError, or
        AdapterError if the two do not merge cleanly."""
        ...

    def symbol_history(
        self, base_sha: str | None, head_sha: str, identifier: str, path: str | None = None
    ) -> list[SymbolChange]:
        """Commits whose diff touches a call/declaration of ``identifier`` (``name(``):
        those reachable from head and those reachable from base since their merge base,
        newest first. Without ``base_sha``, the recent history of head."""
        ...

    def check_patch(self, patch: str, head_sha: str, base_sha: str | None = None) -> PatchCheck:
        """Whether ``patch`` applies to the head tree, or with ``base_sha`` to the merge
        of head into base. Never modifies the repository."""
        ...

    def recent_commits(self, sha: str, limit: int = 10, path: str | None = None) -> list[Commit]:
        """Up to ``limit`` commits reachable from ``sha``, newest first, optionally
        only those touching ``path``."""
        ...


# --------------------------------------------------------------------- Notifier


@dataclass(frozen=True)
class Report:
    """One alert for one failure: one per ``sha + category``."""

    repo: str
    head_sha: str
    build_tools: tuple[BuildTool, ...]  # all builds that failed for this reason
    diagnosis: Diagnosis  # already gated and sanitized by the caller

    @property
    def dedupe_key(self) -> str:
        return f"{self.head_sha}:{self.diagnosis.category.value}"


@runtime_checkable
class Notifier(Protocol):
    def notify(self, report: Report) -> None:
        """Deliver (or update in place) the alert for ``report.dedupe_key``."""
        ...


# ------------------------------------------------------------------ MemoryStore


@dataclass(frozen=True)
class MemoryRecord:
    repo: str
    category: Category
    signature: str  # normalized first fatal error, used for recall
    diagnosis: Diagnosis  # secrets redacted before storing
    confirmed_by: Literal["reaction", "green_build"]  # only confirmed diagnoses are stored
    stored_at: datetime


@dataclass(frozen=True)
class MemoryQuery:
    repo: str
    signature: str | None = None
    category: Category | None = None
    limit: int = 5


@runtime_checkable
class MemoryStore(Protocol):
    def store(self, record: MemoryRecord) -> None:
        """Persist a *confirmed* diagnosis."""
        ...

    def recall(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Past diagnoses as hints only, never as evidence."""
        ...
