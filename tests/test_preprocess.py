import re

import pytest

import cli
from evals.loader import CASES_DIR
from guardrails.preprocess import (
    ErrorKind,
    ExcerptConfig,
    clean_line,
    clean_log,
    find_first_fatal,
    preprocess,
    split_raw_lines,
)

# (case, log) -> (first error line, kind, substring of that line, termination)
# Line numbers are the raw files' own numbering (as `grep -n` prints them).
REAL_LOGS = {
    ("case-01", "maven.log"): (277, ErrorKind.maven_error, "[ERROR] COMPILATION ERROR", None),
    ("case-01", "gradle.log"): (
        259,
        ErrorKind.javac_error,
        "error: no suitable method found for getPet(String,boolean)",
        None,
    ),
    ("case-02", "gradle.log"): (
        239,
        ErrorKind.javac_error,
        "error: package com.google.common.base does not exist",
        None,
    ),
    ("case-03", "maven.log"): (
        4078,
        ErrorKind.out_of_memory,
        "java.lang.OutOfMemoryError: Java heap space",
        None,
    ),
    ("case-03", "gradle.log"): (
        543,
        ErrorKind.out_of_memory,
        "Caused by: java.lang.OutOfMemoryError",
        (550, ErrorKind.cancelled),
    ),
}
# The detail line the model needs, which must also be inside the excerpt window.
KEY_DETAIL = {
    ("case-01", "maven.log"): (
        279,
        "no suitable method found for getPet(java.lang.String,boolean)",
    ),
    # Surefire's fork summary, kept by the tail.
    ("case-03", "maven.log"): (4489, "[ERROR] java.lang.OutOfMemoryError: Java heap space"),
}

NOISE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z|\x1b\[|##\[(?:end)?group\]")


def _raw(case: str, log: str) -> str:
    return (CASES_DIR / case / log).read_text(encoding="utf-8")


# ------------------------------------------------------------------- real logs


@pytest.mark.parametrize(("case", "log"), list(REAL_LOGS))
def test_excerpt_contains_expected_first_error(case: str, log: str) -> None:
    line_no, kind, needle, _ = REAL_LOGS[(case, log)]
    ex = preprocess(_raw(case, log))
    assert ex.first_error is not None
    assert ex.first_error.line.number == line_no
    assert ex.first_error.kind is kind
    assert needle in ex.first_error.line.text
    assert ex.first_error.line in ex.lines
    assert f"{line_no}> " in ex.render()


@pytest.mark.parametrize(("case", "log"), list(KEY_DETAIL))
def test_excerpt_window_includes_key_detail(case: str, log: str) -> None:
    line_no, needle = KEY_DETAIL[(case, log)]
    ex = preprocess(_raw(case, log))
    assert any(line.number == line_no and needle in line.text for line in ex.lines)


def test_case03_gradle_detects_cancellation() -> None:
    ex = preprocess(_raw("case-03", "gradle.log"))
    assert ex.termination is not None
    assert ex.termination.kind is ErrorKind.cancelled
    assert ex.termination.line.number == 550
    assert "The operation was canceled" in ex.termination.line.text
    assert "550! " in ex.render()


@pytest.mark.parametrize(("case", "log"), [k for k in REAL_LOGS if REAL_LOGS[k][3] is None])
def test_no_termination_on_plain_failures(case: str, log: str) -> None:
    assert preprocess(_raw(case, log)).termination is None


@pytest.mark.parametrize(("case", "log"), list(REAL_LOGS))
def test_line_numbers_map_back_to_raw_file(case: str, log: str) -> None:
    raw = _raw(case, log)
    raw_lines = split_raw_lines(raw)
    ex = preprocess(raw)
    # Same count `wc -l` gives, plus one if the file lacks a trailing newline.
    assert ex.total_lines == len(raw_lines) == raw.count("\n") + (not raw.endswith("\n"))
    for line in ex.lines:
        assert clean_line(raw_lines[line.number - 1]) == line.text
    first = ex.first_error
    assert first is not None
    # Cleaning only removes prefixes and trailing whitespace from plain lines.
    assert raw_lines[first.line.number - 1].rstrip().endswith(first.line.text)


@pytest.mark.parametrize(("case", "log"), list(REAL_LOGS))
def test_excerpt_is_clean_and_smaller(case: str, log: str) -> None:
    raw = _raw(case, log)
    ex = preprocess(raw)
    rendered = ex.render()
    assert not NOISE.search(rendered)
    assert "﻿" not in rendered
    assert len(ex.lines) < ex.total_lines
    # Nothing after the job's last ##[error] (post-job cleanup) is kept.
    last_error = max(i for i, t in enumerate(split_raw_lines(raw), 1) if "##[error]" in t)
    assert max(line.number for line in ex.lines) == last_error


def test_large_log_is_cut_hard() -> None:
    ex = preprocess(_raw("case-03", "maven.log"))
    cfg = ExcerptConfig()
    assert ex.total_lines > 4000
    assert len(ex.lines) <= cfg.head + cfg.before + cfg.after + 1 + cfg.tail


# ---------------------------------------------------------------- unit: clean


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-11-08T13:32:30.1826870Z [ERROR] boom", "[ERROR] boom"),
        ("﻿2026-11-07T23:36:20.6181564Z Current runner version", "Current runner version"),
        ("2026-11-08T13:32:22.7340937Z \x1b[36;1m./mvnw -B verify\x1b[0m", "./mvnw -B verify"),
        ("2026-11-08T13:32:22.7340689Z ##[group]Run ./mvnw -B verify", "Run ./mvnw -B verify"),
        ("2026-11-08T13:32:22.7340689Z ##[endgroup]", None),
        ("build (17)\tRun ./gradlew build\t2026-11-08T13:32:36.6Z FAILURE: x", "FAILURE: x"),
        (
            "2026-11-07T23:37:20.8Z 2026-11-07T23:37:20.695Z  WARN 2688",
            "2026-11-07T23:37:20.695Z  WARN 2688",
        ),
        ("no prefix at all\r", "no prefix at all\r".rstrip()),
    ],
)
def test_clean_line(raw: str, expected: str | None) -> None:
    assert clean_line(raw) == expected


def test_split_raw_lines_matches_grep_numbering() -> None:
    assert split_raw_lines("a\r\nb\x0cc\nd\n") == ["a", "b\x0cc", "d"]
    assert split_raw_lines("a\nb") == ["a", "b"]


# --------------------------------------------------------------- unit: detect


def _lines(*texts: str) -> str:
    return "".join(f"2026-11-08T00:00:{i:02d}.0000000Z {t}\n" for i, t in enumerate(texts))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("[ERROR] Failed to execute goal x", ErrorKind.maven_error),
        ("/src/A.java:12: error: cannot find symbol", ErrorKind.javac_error),
        ("FAILURE: Build failed with an exception.", ErrorKind.gradle_failure),
        ("* What went wrong:", ErrorKind.gradle_failure),
        ("java.lang.OutOfMemoryError: GC overhead limit exceeded", ErrorKind.out_of_memory),
        ("##[error]The operation was canceled.", ErrorKind.cancelled),
        (
            "##[error]The job running on runner X has exceeded the maximum execution time"
            " of 15 minutes.",
            ErrorKind.timeout,
        ),
    ],
)
def test_detects_each_fatal_kind(text: str, kind: ErrorKind) -> None:
    lines, _ = clean_log(_lines("[INFO] ok", text))
    first = find_first_fatal(lines)
    assert first is not None and first.kind is kind and first.line.number == 2


@pytest.mark.parametrize(
    "text",
    [
        "[ERROR] ",  # Maven's blank separator line
        "[INFO] BUILD FAILURE is not a fatal pattern on its own",
        "2026-11-07T23:37:20.695Z ERROR 2688 --- [main] o.s.Foo : spring log line",
        "##[error]Process completed with exit code 1.",
        "-Werror passed to javac",
    ],
)
def test_ignores_non_fatal_lines(text: str) -> None:
    lines, _ = clean_log(_lines(text))
    assert find_first_fatal(lines) is None


def test_first_error_wins_over_later_ones() -> None:
    raw = _lines("start", "/A.java:1: error: x", "FAILURE: Build failed", "OutOfMemoryError")
    ex = preprocess(raw)
    assert ex.first_error is not None and ex.first_error.line.number == 2


def test_no_fatal_error_gives_head_and_tail_only() -> None:
    raw = _lines(*[f"line {i}" for i in range(200)])
    ex = preprocess(raw, ExcerptConfig(head=5, tail=5))
    assert ex.first_error is None and ex.termination is None
    assert [line.number for line in ex.lines] == [1, 2, 3, 4, 5, 196, 197, 198, 199, 200]
    assert "[190 lines omitted]" in ex.render()


# ------------------------------------------------------------------------- cli


def test_cli_preprocess_prints_excerpt(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["preprocess", "--case", "case-01", "--log", "maven.log"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# case-01/maven.log: 324 raw lines")
    assert "first fatal error: maven_error at line 277" in out
    assert "277> [ERROR] COMPILATION ERROR" in out


def test_cli_unknown_log_is_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["preprocess", "--case", "case-02", "--log", "maven.log"]) == 2
    assert "error:" in capsys.readouterr().err
