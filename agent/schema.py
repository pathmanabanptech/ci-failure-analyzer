"""Output contract. Every agent returns a ``Diagnosis``."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SpecialistName = Literal["build", "test", "infra"]
BuildTool = Literal["maven", "gradle"]


class Category(str, Enum):  # noqa: UP042  (str mixin is part of the contract)
    compile = "compile"
    dependency = "dependency"
    # Shadows str.format on members; intentional, the value "format" is the contract.
    format = "format"  # type: ignore[assignment]
    test_assertion = "test_assertion"
    flaky = "flaky"
    memory = "memory"
    config = "config"
    dependency_service = "dependency_service"
    timeout = "timeout"
    runner = "runner"
    cancelled = "cancelled"
    unknown = "unknown"


class Status(str, Enum):  # noqa: UP042  (str mixin is part of the contract)
    diagnosed = "diagnosed"
    possible_cause = "possible_cause"
    undiagnosable = "undiagnosable"


EvidenceKind = Literal["log", "file", "commit", "run_attempts"]
Ref = Literal["head", "base", "merge"]  # merge: head merged into base (what a PR build compiles)


class Evidence(BaseModel):
    """One cited fact. The ``snippet`` is the evidence; ``line`` is only a hint.

    Which fields apply depends on ``kind``:

    * ``log``: ``log`` names the job log (default: the log being analyzed).
    * ``file``: ``path`` (repo-relative) and ``ref`` (default: ``merge`` for
      pull_request builds, ``head`` otherwise).
    * ``commit``: ``sha`` (full or prefix) of a commit the agent was shown.
    * ``run_attempts``: no locator.

    Missing or inconsistent locators are not a schema error: the evidence verifier
    rejects such evidence with a reason.
    """

    kind: EvidenceKind
    log: str | None = None
    path: str | None = None
    ref: Ref | None = None
    sha: str | None = None
    line: int | None = None  # the model's hint (1-based line in the log or file)
    snippet: str
    located_line: int | None = None  # where the verifier found the snippet; verifier only
    verified: bool = False  # set by the evidence verifier, never by the model

    def where(self) -> str:
        """Short human-readable location, e.g. ``maven.log:279`` or ``pom.xml@merge:40``."""
        line = self.located_line if self.located_line is not None else self.line
        at = f":{line}" if line is not None else ""
        if self.kind == "log":
            return f"{self.log or 'log'}{at}"
        if self.kind == "file":
            return f"{self.path or '?'}@{self.ref or 'default'}{at}"
        if self.kind == "commit":
            return f"commit {(self.sha or '?')[:12]}"
        return "run_attempts"


class Diagnosis(BaseModel):
    status: Status
    category: Category
    specialist: SpecialistName | None
    root_cause: str  # one or two sentences
    local_vs_ci: str  # why it probably passed on the dev's machine
    evidence: list[Evidence]
    fix_diff: str | None  # minimal suggested change
    confidence: float = Field(
        ge=0.0, le=1.0
    )  # model-reported, then capped by the evidence verifier
    reroute: SpecialistName | None = None
    build_tool: BuildTool | None = None
    auto_fixable: bool = False  # set by auto-fix gating code, never by the model
