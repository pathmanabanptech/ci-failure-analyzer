"""Offline consistency checks: does each case's meta.yaml match its own logs?

Catches the hand-curation mistakes that hit the Phase 0 cases: logs filed under the
wrong case, a log copied into two cases, run/job ids copy-pasted between runs, and a
conclusion that disagrees with the log (cancelled vs failure).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from evals.loader import CASES_DIR, list_cases, load_meta
from evals.log_facts import extract_log_facts
from guardrails.preprocess import ErrorKind


def check_case(case_id: str, cases_dir: Path = CASES_DIR) -> list[str]:
    """Problems found in one case (empty list = consistent)."""
    meta = load_meta(case_id, cases_dir)
    problems: list[str] = []

    if meta.event == "pull_request" and meta.base_sha is None:
        problems.append(f"{case_id}: pull_request case has no base_sha")

    run_ids: dict[int, str] = {}
    job_ids: dict[int, str] = {}
    for tool, run in meta.runs.items():
        if run.run_id in run_ids:
            problems.append(f"{case_id}: runs.{tool} reuses run_id of runs.{run_ids[run.run_id]}")
        run_ids[run.run_id] = tool
        if run.job_id is not None:
            if run.job_id in job_ids:
                problems.append(
                    f"{case_id}: runs.{tool} reuses job_id of runs.{job_ids[run.job_id]}"
                )
            job_ids[run.job_id] = tool
    for rel in meta.related_runs:
        if rel.run_id in run_ids:
            problems.append(f"{case_id}: related run {rel.run_id} is also a captured run")

    for run in meta.runs.values():
        if run.log is None:
            continue
        where = f"{case_id}/{run.log}"
        facts = extract_log_facts((cases_dir / case_id / run.log).read_text(encoding="utf-8"))

        if meta.event == "pull_request":
            if facts.merge_head is None:
                problems.append(f"{where}: no 'Merge <head> into <base>' checkout line")
            else:
                if facts.merge_head != meta.head_sha:
                    problems.append(
                        f"{where}: log built head {facts.merge_head[:12]}, "
                        f"meta head_sha is {meta.head_sha[:12]}"
                    )
                if meta.base_sha and facts.merge_base != meta.base_sha:
                    problems.append(
                        f"{where}: log merged into {(facts.merge_base or '?')[:12]}, "
                        f"meta base_sha is {meta.base_sha[:12]}"
                    )
        elif facts.checked_out_sha != meta.head_sha:
            got = (facts.checked_out_sha or "nothing")[:12]
            problems.append(
                f"{where}: log checked out {got}, meta head_sha is {meta.head_sha[:12]}"
            )

        if facts.java_version is not None and facts.java_version != meta.jdk:
            problems.append(f"{where}: log java-version {facts.java_version}, meta jdk {meta.jdk}")

        if facts.termination is ErrorKind.cancelled and run.conclusion not in (
            "cancelled",
            "timed_out",
        ):
            problems.append(f"{where}: log shows the job was cancelled, meta says {run.conclusion}")
        if facts.termination is None and run.conclusion in ("cancelled", "timed_out"):
            problems.append(f"{where}: meta says {run.conclusion}, log has no cancel/timeout")

    return problems


def check_all_cases(cases_dir: Path = CASES_DIR) -> list[str]:
    """Per-case checks plus cross-case duplicates (identical logs, reused ids)."""
    problems: list[str] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_run: dict[int, list[str]] = defaultdict(list)
    by_job: dict[int, list[str]] = defaultdict(list)

    for case_id in list_cases(cases_dir):
        problems.extend(check_case(case_id, cases_dir))
        meta = load_meta(case_id, cases_dir)
        for tool, run in meta.runs.items():
            by_run[run.run_id].append(f"{case_id}/{tool}")
            if run.job_id is not None:
                by_job[run.job_id].append(f"{case_id}/{tool}")
            if run.log is not None:
                digest = hashlib.sha256((cases_dir / case_id / run.log).read_bytes()).hexdigest()
                by_hash[digest].append(f"{case_id}/{run.log}")

    for files in by_hash.values():
        if len(files) > 1:
            problems.append(f"identical logs: {', '.join(files)}")
    for kind, index in (("run_id", by_run), ("job_id", by_job)):
        for ident, users in index.items():
            # Same-case reuse is already reported by check_case.
            if len({u.split("/")[0] for u in users}) > 1:
                problems.append(f"{kind} {ident} used by {', '.join(users)}")
    return problems
