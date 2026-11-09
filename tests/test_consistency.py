import shutil
from pathlib import Path

import pytest

from evals.consistency import check_all_cases, check_case
from evals.loader import CASES_DIR, list_cases
from evals.log_facts import extract_log_facts
from guardrails.preprocess import ErrorKind

HEAD = "a" * 40
BASE = "b" * 40
OTHER = "c" * 40


def push_log(sha: str = HEAD, java: str = "17", tail: str = "##[error]Process completed.") -> str:
    return (
        "﻿2026-11-08T00:00:00.0000000Z ##[group]Run actions/checkout@v4\n"
        "2026-11-08T00:00:01.0000000Z [command]/usr/bin/git log -1 --format=%H\n"
        f"2026-11-08T00:00:02.0000000Z {sha}\n"
        "2026-11-08T00:00:03.0000000Z ##[group]Run actions/setup-java@v4\n"
        f"2026-11-08T00:00:04.0000000Z   java-version: {java}\n"
        "2026-11-08T00:00:05.0000000Z [ERROR] boom\n"
        f"2026-11-08T00:00:06.0000000Z {tail}\n"
    )


def pr_log(head: str = HEAD, base: str = BASE) -> str:
    return (
        f"2026-11-08T00:00:00.0000000Z HEAD is now at 0b5a2fb Merge {head} into {base}\n"
        "2026-11-08T00:00:01.0000000Z   java-version: 17\n"
        "2026-11-08T00:00:02.0000000Z [ERROR] boom\n"
    )


def write_case(
    root: Path,
    case_id: str,
    logs: dict[str, str],
    *,
    event: str = "push",
    base: str | None = None,
    ids: dict[str, tuple[int, int]] | None = None,
    conclusion: str = "failure",
    jdk: str = "17",
) -> None:
    d = root / case_id
    d.mkdir()
    ids = ids or {"maven": (1, 11), "gradle": (2, 22)}
    lines = [
        "repo: o/r",
        f"event: {event}",
        f"head_sha: {HEAD}",
        *([f"base_sha: {base}"] if base else []),
        "runner_os: ubuntu-latest",
        f'jdk: "{jdk}"',
        "runs:",
    ]
    for tool, text in logs.items():
        run_id, job_id = ids[tool]
        (d / f"{tool}.log").write_text(text)
        lines.append(
            f"  {tool}: {{ run_id: {run_id}, job_id: {job_id}, conclusion: {conclusion},"
            f" attempts: 1, log: {tool}.log }}"
        )
    (d / "meta.yaml").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------- real cases


def test_real_cases_are_consistent() -> None:
    assert check_all_cases() == []


@pytest.mark.parametrize("case_id", list_cases())
def test_real_case_facts(case_id: str) -> None:
    assert check_case(case_id) == []


def test_real_pr_case_merge_line() -> None:
    raw = (CASES_DIR / "case-01" / "maven.log").read_text(encoding="utf-8")
    facts = extract_log_facts(raw)
    assert facts.merge_head == "c9d5394f2b069c2b80fc6d1fb18340accfc6f069"
    assert facts.merge_base == "afab46b773c6b540fad0e470685afff76f7363a3"
    assert facts.java_version == "17"


def test_real_cancelled_case_facts() -> None:
    raw = (CASES_DIR / "case-03" / "gradle.log").read_text(encoding="utf-8")
    facts = extract_log_facts(raw)
    assert facts.checked_out_sha == "259023cc03681eaaae64039c9e7ea1244520e403"
    assert facts.termination is ErrorKind.cancelled


# ------------------------------------------------------ synthetic broken cases


def test_consistent_synthetic_case(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": push_log(), "gradle": push_log(tail="FAILURE: x")})
    assert check_all_cases(tmp_path) == []


def test_log_filed_under_wrong_case(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": push_log(sha=OTHER)})
    [problem] = check_case("case-x", tmp_path)
    assert "checked out cccccccccccc" in problem


def test_pr_log_with_wrong_base(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": pr_log(base=OTHER)}, event="pull_request", base=BASE)
    [problem] = check_case("case-x", tmp_path)
    assert "merged into cccccccccccc" in problem


def test_pr_case_without_base(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": pr_log()}, event="pull_request")
    assert any("no base_sha" in p for p in check_case("case-x", tmp_path))


def test_copy_pasted_ids(tmp_path: Path) -> None:
    write_case(
        tmp_path,
        "case-x",
        {"maven": push_log(), "gradle": push_log("a" * 40, "17", "x")},
        ids={"maven": (7, 70), "gradle": (7, 70)},
    )
    problems = check_case("case-x", tmp_path)
    assert any("reuses run_id" in p for p in problems)
    assert any("reuses job_id" in p for p in problems)


def test_same_log_in_two_cases(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": push_log()})
    write_case(tmp_path, "case-y", {"maven": push_log()}, ids={"maven": (3, 33)})
    assert any(p.startswith("identical logs:") for p in check_all_cases(tmp_path))


def test_ids_reused_across_cases(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": push_log()})
    write_case(tmp_path, "case-y", {"maven": push_log(tail="other")})
    problems = check_all_cases(tmp_path)
    assert any("run_id 1 used by case-x/maven, case-y/maven" in p for p in problems)


def test_cancelled_log_but_meta_says_failure(tmp_path: Path) -> None:
    write_case(
        tmp_path, "case-x", {"gradle": push_log(tail="##[error]The operation was canceled.")}
    )
    [problem] = check_case("case-x", tmp_path)
    assert "cancelled, meta says failure" in problem


def test_meta_says_cancelled_but_log_does_not(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"gradle": push_log()}, conclusion="cancelled")
    [problem] = check_case("case-x", tmp_path)
    assert "no cancel/timeout" in problem


def test_jdk_mismatch(tmp_path: Path) -> None:
    write_case(tmp_path, "case-x", {"maven": push_log(java="21")})
    [problem] = check_case("case-x", tmp_path)
    assert "java-version 21, meta jdk 17" in problem


def test_pre_fix_phase0_data_is_flagged(tmp_path: Path) -> None:
    """The original hand-made layout: maven logs swapped, gradle log duplicated."""
    for c in ("case-01", "case-03"):
        shutil.copytree(CASES_DIR / c, tmp_path / c)
    swap = tmp_path / "swap.tmp"
    (tmp_path / "case-01/maven.log").rename(swap)
    (tmp_path / "case-03/maven.log").rename(tmp_path / "case-01/maven.log")
    swap.rename(tmp_path / "case-03/maven.log")
    shutil.copy(tmp_path / "case-01/gradle.log", tmp_path / "case-03/gradle.log")

    problems = check_all_cases(tmp_path)
    assert any("case-01/maven.log" in p for p in problems)
    assert any("case-03/maven.log" in p for p in problems)
    assert any("identical logs: case-01/gradle.log, case-03/gradle.log" in p for p in problems)
