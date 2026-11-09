"""Output contract (PLAN.md section 6). Every agent returns a ``Diagnosis``."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SpecialistName = Literal["build", "test", "infra"]
BuildTool = Literal["maven", "gradle"]


class Category(str, Enum):  # noqa: UP042  (contract in PLAN.md section 6)
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


class Status(str, Enum):  # noqa: UP042  (contract in PLAN.md section 6)
    diagnosed = "diagnosed"
    possible_cause = "possible_cause"
    undiagnosable = "undiagnosable"


class Evidence(BaseModel):
    source: str  # "log" | file path | "commit" | "run_attempts"
    line: int | None = None
    snippet: str
    verified: bool = False  # set by G5 verifier, never by the model


class Diagnosis(BaseModel):
    status: Status
    category: Category
    specialist: SpecialistName | None
    root_cause: str  # one or two sentences
    local_vs_ci: str  # why it probably passed on the dev's machine
    evidence: list[Evidence]
    fix_diff: str | None  # minimal suggested change
    confidence: float = Field(ge=0.0, le=1.0)  # model-reported, then capped by G5
    reroute: SpecialistName | None = None
    build_tool: BuildTool | None = None
    copilot_fixable: bool = False  # set by G11 gating code, never by the model
