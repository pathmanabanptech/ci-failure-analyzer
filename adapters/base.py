"""The four I/O interfaces (design rule 1). Agent logic talks only to these.

Phase 1 ships local implementations; Phase 3 adds GitHub ones and Phase 4 AWS ones
behind the same Protocols, so agent code, prompts and evals do not change.

Content returned by ``LogSource`` and ``RepoReader`` (logs, file contents, commit
messages) is untrusted (G1). Adapters return it raw; callers redact (G3) and wrap it
in untrusted blocks before it reaches a model.
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
    subject: str  # untrusted (G1); no author name/email (not needed, PII)


@runtime_checkable
class RepoReader(Protocol):
    def read_file(self, sha: str, path: str) -> str:
        """File content at a commit. Raises AccessDeniedError / NotFoundError."""
        ...

    def recent_commits(self, sha: str, limit: int = 10, path: str | None = None) -> list[Commit]:
        """Up to ``limit`` commits reachable from ``sha``, newest first, optionally
        only those touching ``path``."""
        ...


# --------------------------------------------------------------------- Notifier


@dataclass(frozen=True)
class Report:
    """One alert for one failure (G10: one per ``sha + category``)."""

    repo: str
    head_sha: str
    build_tools: tuple[BuildTool, ...]  # all builds that failed for this reason
    diagnosis: Diagnosis  # already gated (G9/G11) and sanitized (G2) by the caller

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
    diagnosis: Diagnosis  # redacted (G3) before storing
    confirmed_by: Literal["reaction", "green_build"]  # G12: only confirmed diagnoses
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
        """Persist a *confirmed* diagnosis (G12)."""
        ...

    def recall(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Past diagnoses as hints only. Never evidence (G12, G5)."""
        ...
