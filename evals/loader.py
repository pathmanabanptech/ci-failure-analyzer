"""Eval case loader.

Two strictly separated views of a case:

* ``AgentInput`` (from ``load_agent_input``): what the agent may see — the case's
  ``meta.yaml`` and which log to analyze. It never reads ``expected.yaml``.
* ``AnswerKey`` (from ``load_answer_key``): the expected result for one log, with
  ``by_log`` overrides applied. Only callable from code under ``evals/`` or
  ``tests/``; any other caller gets ``AnswerKeyAccessError``.

The ``meta.yaml`` model and its validation live in ``adapters/logs/case_files.py``
(the local log source reads them too) and are re-exported here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from adapters.logs import case_files
from adapters.logs.case_files import (
    FORBIDDEN_META_KEYS,
    CaseError,
    CaseMeta,
    Conclusion,
    Event,
    RelatedRun,
    RunInfo,
    Sha,
    find_forbidden_keys,
)
from agent.schema import BuildTool, Category, SpecialistName, Status

__all__ = [
    "CASES_DIR",
    "FORBIDDEN_META_KEYS",
    "AgentInput",
    "AnswerKey",
    "AnswerKeyAccessError",
    "CaseError",
    "CaseMeta",
    "Conclusion",
    "Event",
    "RelatedRun",
    "RunInfo",
    "Sha",
    "case_logs",
    "find_forbidden_keys",
    "list_cases",
    "load_agent_input",
    "load_answer_key",
    "load_meta",
]

EVALS_DIR = Path(__file__).resolve().parent
CASES_DIR = EVALS_DIR / "cases"
_REPO_ROOT = EVALS_DIR.parent
_ANSWER_KEY_CALLER_DIRS = (EVALS_DIR, _REPO_ROOT / "tests")


class AnswerKeyAccessError(PermissionError):
    """expected.yaml was requested from outside evals/ (or tests/)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


def list_cases(cases_dir: Path = CASES_DIR) -> list[str]:
    return case_files.list_cases(cases_dir)


def load_meta(case_id: str, cases_dir: Path = CASES_DIR) -> CaseMeta:
    return case_files.load_meta(case_id, cases_dir)


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
            case_files.read_yaml(case_files.case_dir(case_id, cases_dir) / "expected.yaml")
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
