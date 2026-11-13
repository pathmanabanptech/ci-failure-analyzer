"""LogSource over the local eval cases (``evals/cases/<case>/``).

Runs are addressed by ``RunRef(repo, run_id)`` exactly as the GitHub source will
address them; the case is found by looking the run id up in the cases' meta.yaml.
"""

from __future__ import annotations

from pathlib import Path

from adapters.base import NotFoundError, RelatedRun, RunAttempt, RunDetails, RunRef
from adapters.logs.case_files import CaseMeta, RunInfo, list_cases, load_meta
from agent.schema import BuildTool


class LocalLogSource:
    def __init__(self, cases_dir: Path) -> None:
        self._cases_dir = cases_dir

    def ref_for(self, case_id: str, log: str) -> RunRef:
        """The RunRef of the run whose log file is ``log`` in ``case_id``."""
        meta = load_meta(case_id, self._cases_dir)
        for run in meta.runs.values():
            if run.log == log:
                return RunRef(repo=meta.repo, run_id=run.run_id, job_id=run.job_id)
        raise NotFoundError(f"{case_id} has no log {log!r}")

    def _find(self, ref: RunRef) -> tuple[str, CaseMeta, BuildTool, RunInfo]:
        for case_id in list_cases(self._cases_dir):
            meta = load_meta(case_id, self._cases_dir)
            if meta.repo != ref.repo:
                continue
            for tool, run in meta.runs.items():
                if run.run_id == ref.run_id and ref.job_id in (None, run.job_id):
                    return case_id, meta, tool, run
        raise NotFoundError(f"no local case has run {ref.run_id} of {ref.repo}")

    def get_log(self, run: RunRef) -> str:
        case_id, _, _, info = self._find(run)
        if info.log is None:
            raise NotFoundError(f"run {run.run_id} succeeded; it has no failure log")
        return (self._cases_dir / case_id / info.log).read_text(encoding="utf-8")

    def get_run(self, run: RunRef) -> RunDetails:
        _, meta, tool, info = self._find(run)
        # meta records the attempt count and the latest conclusion only.
        attempts = tuple(
            RunAttempt(attempt=n, conclusion=info.conclusion if n == info.attempts else None)
            for n in range(1, info.attempts + 1)
        )
        return RunDetails(
            ref=RunRef(repo=meta.repo, run_id=info.run_id, job_id=info.job_id),
            build_tool=tool,
            event=meta.event,
            head_sha=meta.head_sha,
            base_sha=meta.base_sha,
            conclusion=info.conclusion,
            runner_os=meta.runner_os,
            attempts=attempts,
            related_runs=tuple(
                RelatedRun(
                    build_tool=r.tool,
                    run_id=r.run_id,
                    event=r.event,
                    conclusion=r.conclusion,
                    base_sha=r.base_sha,
                )
                for r in meta.related_runs
            ),
        )
