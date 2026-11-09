"""Eval case loader.

Two strictly separated views of a case:

* ``AgentInput`` (from ``load_agent_input``): what the agent may see — the case's
  ``meta.yaml`` and which log to analyze. It never reads ``expected.yaml``.
* ``AnswerKey`` (from ``load_answer_key``): the expected result for one log, with
  ``by_log`` overrides applied. Only callable from code under ``evals/`` or
  ``tests/``; any other caller gets ``AnswerKeyAccessError``.

``meta.yaml`` is validated with ``extra="forbid"`` and a deny-list of label-leaking
keys (branch names, PR titles), so a leak in the case data fails loudly at load time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.schema import BuildTool, Category, SpecialistName, Status

EVALS_DIR = Path(__file__).resolve().parent
CASES_DIR = EVALS_DIR / "cases"
_REPO_ROOT = EVALS_DIR.parent
_ANSWER_KEY_CALLER_DIRS = (EVALS_DIR, _REPO_ROOT / "tests")

# Keys that would leak labels into agent input (G4 / section 7.3 "no label leakage").
FORBIDDEN_META_KEYS = frozenset(
    {
        "branch",
        "head_branch",
        "base_branch",
        "ref",
        "head_ref",
        "base_ref",
        "pr_title",
        "title",
        "pr_body",
        "description",
    }
)


class CaseError(ValueError):
    """A case directory is missing files or contains invalid data."""


class AnswerKeyAccessError(PermissionError):
    """expected.yaml was requested from outside evals/ (or tests/)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------- meta.yaml


class RunInfo(_Strict):
    run_id: int
    job_id: int | None = None
    conclusion: Literal["success", "failure", "cancelled", "timed_out"]
    attempts: int = Field(ge=1)
    log: str | None = None

    @model_validator(mode="after")
    def _failed_runs_have_logs(self) -> RunInfo:
        if self.conclusion != "success" and self.log is None:
            raise ValueError(f"run {self.run_id} did not succeed but has no log")
        return self


class RelatedRun(_Strict):
    run_id: int
    conclusion: Literal["success", "failure", "cancelled", "timed_out"]


class CaseMeta(_Strict):
    repo: str
    event: Literal["push", "pull_request", "workflow_dispatch"]
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    runner_os: str
    jdk: str
    runs: dict[BuildTool, RunInfo]
    related_runs: dict[BuildTool, RelatedRun] | None = None

    @field_validator("repo")
    @classmethod
    def _normalize_repo(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        owner, sep, name = v.partition("/")
        if not sep or not owner or not name or "/" in name:
            raise ValueError(f"repo must be 'owner/name', got {v!r}")
        return v


class AgentInput(_Strict):
    """Everything the agent is allowed to know about a case. No answer fields."""

    case_id: str
    log: str
    meta: CaseMeta


# ----------------------------------------------------------------- expected.yaml


class _ExpectedFields(_Strict):
    category: Category | None = None
    specialist: SpecialistName | None = None
    status: Status | None = None
    build_tool: BuildTool | None = None
    evidence_sources: list[str] | None = None
    key_facts: list[str] | None = None
    fix_should_build: bool | None = None
    expected_confidence: tuple[float, float] | None = None
    dedupe: bool | None = None
    tags: list[str] | None = None


class _ExpectedFile(_ExpectedFields):
    by_log: dict[str, _ExpectedFields] = Field(default_factory=dict)


class AnswerKey(_Strict):
    """Resolved answer key for one (case, log) pair (top-level defaults + by_log override)."""

    case_id: str
    log: str
    category: Category
    specialist: SpecialistName | None
    status: Status
    build_tool: BuildTool | None = None
    evidence_sources: list[str]
    key_facts: list[str]
    fix_should_build: bool | None = None
    expected_confidence: tuple[float, float] | None = None
    dedupe: bool = False
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------- loading


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CaseError(f"missing {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise CaseError(f"{path} must contain a mapping")
    return data


def _find_forbidden_keys(node: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k).lower() in FORBIDDEN_META_KEYS:
                found.append(here)
            found.extend(_find_forbidden_keys(v, here))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.extend(_find_forbidden_keys(v, f"{path}[{i}]"))
    return found


def _case_dir(case_id: str, cases_dir: Path) -> Path:
    if "/" in case_id or "\\" in case_id or case_id in {"", ".", ".."}:
        raise CaseError(f"invalid case id {case_id!r}")
    d = cases_dir / case_id
    if not d.is_dir():
        raise CaseError(f"no such case: {d}")
    return d


def list_cases(cases_dir: Path = CASES_DIR) -> list[str]:
    return sorted(p.name for p in cases_dir.iterdir() if (p / "meta.yaml").is_file())


def load_meta(case_id: str, cases_dir: Path = CASES_DIR) -> CaseMeta:
    d = _case_dir(case_id, cases_dir)
    raw = _read_yaml(d / "meta.yaml")
    leaks = _find_forbidden_keys(raw)
    if leaks:
        raise CaseError(f"{case_id}/meta.yaml contains label-leaking keys: {leaks}")
    try:
        meta = CaseMeta.model_validate(raw)
    except ValueError as e:
        raise CaseError(f"{case_id}/meta.yaml: {e}") from e
    for tool, run in meta.runs.items():
        if run.log is not None and not (d / run.log).is_file():
            raise CaseError(f"{case_id}: runs.{tool}.log points to missing file {run.log}")
    return meta


def case_logs(case_id: str, cases_dir: Path = CASES_DIR) -> list[str]:
    """Log file names of the case's failed runs, in meta order."""
    return [r.log for r in load_meta(case_id, cases_dir).runs.values() if r.log is not None]


def load_agent_input(case_id: str, log: str, cases_dir: Path = CASES_DIR) -> AgentInput:
    meta = load_meta(case_id, cases_dir)
    logs = [r.log for r in meta.runs.values() if r.log is not None]
    if log not in logs:
        raise CaseError(f"{case_id} has no log {log!r} (available: {logs})")
    return AgentInput(case_id=case_id, log=log, meta=meta)


def _check_answer_key_caller() -> None:
    # Frame 0: this function, 1: load_answer_key, 2: the caller.
    caller = Path(sys._getframe(2).f_code.co_filename).resolve()
    if not any(caller.is_relative_to(d) for d in _ANSWER_KEY_CALLER_DIRS):
        raise AnswerKeyAccessError(
            f"expected.yaml may only be read from evals/ or tests/, not from {caller}"
        )


def load_answer_key(case_id: str, log: str, cases_dir: Path = CASES_DIR) -> AnswerKey:
    _check_answer_key_caller()
    logs = case_logs(case_id, cases_dir)
    if log not in logs:
        raise CaseError(f"{case_id} has no log {log!r} (available: {logs})")
    try:
        expected = _ExpectedFile.model_validate(
            _read_yaml(_case_dir(case_id, cases_dir) / "expected.yaml")
        )
    except ValueError as e:
        raise CaseError(f"{case_id}/expected.yaml: {e}") from e
    unknown = set(expected.by_log) - set(logs)
    if unknown:
        raise CaseError(f"{case_id}/expected.yaml by_log names unknown logs: {sorted(unknown)}")

    merged = expected.model_dump(exclude={"by_log"}, exclude_none=True)
    if log in expected.by_log:
        merged.update(expected.by_log[log].model_dump(exclude_none=True))
    try:
        return AnswerKey(case_id=case_id, log=log, **merged)
    except ValueError as e:
        raise CaseError(f"{case_id}/expected.yaml ({log}): {e}") from e
