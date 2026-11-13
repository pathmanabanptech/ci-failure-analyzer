"""On-disk case format (``evals/cases/<case>/meta.yaml`` + raw logs).

Lives in ``adapters/`` because ``LocalLogSource`` reads it; ``evals/loader.py``
re-exports it. Only ``meta.yaml`` and logs are handled here; the answer key is
handled solely by ``evals/`` and never imported by agent-side packages.

``meta.yaml`` is validated with ``extra="forbid"`` and a deny-list of label-leaking
keys (branch names, PR titles), so a leak in the case data fails loudly at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.schema import BuildTool

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


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Conclusion = Literal["success", "failure", "cancelled", "timed_out"]
Event = Literal["push", "pull_request", "workflow_dispatch"]
Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class RunInfo(_Strict):
    run_id: int
    job_id: int | None = None
    conclusion: Conclusion
    attempts: int = Field(ge=1)
    log: str | None = None

    @model_validator(mode="after")
    def _failed_runs_have_logs(self) -> RunInfo:
        if self.conclusion != "success" and self.log is None:
            raise ValueError(f"run {self.run_id} did not succeed but has no log")
        return self


class RelatedRun(_Strict):
    """Another run of the same head_sha (e.g. the branch build that passed)."""

    tool: BuildTool
    run_id: int
    event: Event
    conclusion: Conclusion
    base_sha: Sha | None = None  # PR runs: the base the merge was built against


class CaseMeta(_Strict):
    repo: str
    event: Event
    head_sha: Sha
    base_sha: Sha | None = None
    runner_os: str
    jdk: str
    runs: dict[BuildTool, RunInfo]
    related_runs: list[RelatedRun] = Field(default_factory=list)

    @field_validator("repo")
    @classmethod
    def _normalize_repo(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        owner, sep, name = v.partition("/")
        if not sep or not owner or not name or "/" in name:
            raise ValueError(f"repo must be 'owner/name', got {v!r}")
        return v


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CaseError(f"missing {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise CaseError(f"{path} must contain a mapping")
    return data


def find_forbidden_keys(node: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k).lower() in FORBIDDEN_META_KEYS:
                found.append(here)
            found.extend(find_forbidden_keys(v, here))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.extend(find_forbidden_keys(v, f"{path}[{i}]"))
    return found


def case_dir(case_id: str, cases_dir: Path) -> Path:
    if "/" in case_id or "\\" in case_id or case_id in {"", ".", ".."}:
        raise CaseError(f"invalid case id {case_id!r}")
    d = cases_dir / case_id
    if not d.is_dir():
        raise CaseError(f"no such case: {d}")
    return d


def list_cases(cases_dir: Path) -> list[str]:
    return sorted(p.name for p in cases_dir.iterdir() if (p / "meta.yaml").is_file())


def load_meta(case_id: str, cases_dir: Path) -> CaseMeta:
    d = case_dir(case_id, cases_dir)
    raw = read_yaml(d / "meta.yaml")
    leaks = find_forbidden_keys(raw)
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
