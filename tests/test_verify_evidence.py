"""Evidence verifier: the snippet is the evidence, the line is a hint.

Real case-01 maven.log facts used throughout: line 277 is ``[ERROR] COMPILATION ERROR :``
(the first fatal error); the compiler message ``REAL`` is at lines 279 and 293.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from agent.schema import Diagnosis, Ref, Status
from evals.loader import CASES_DIR
from guardrails.preprocess import clean_log, preprocess, split_raw_lines
from guardrails.verify_evidence import (
    COMMIT_PREFIX,
    LogView,
    default_ref_for,
    locate,
    verify_evidence,
)
from pipeline import redact_excerpt

REAL = "Owner.java:[194,24] no suitable method found for getPet(java.lang.String,boolean)"
OWNER = "src/main/java/org/springframework/samples/petclinic/owner/Owner.java"
HEAD_SRC = "\n".join(
    [
        "class Owner {",
        "    boolean hasPet(String n) {",
        "        return getPet(n, true) != null;",
        "    }",
        "}",
    ]
)
MERGE_SRC = HEAD_SRC.replace("class Owner {", "class Owner {\n    Pet findPetByName(String n) {}")
FILES: dict[Ref, dict[str, str]] = {"head": {OWNER: HEAD_SRC}, "merge": {OWNER: MERGE_SRC}}


def read(path: str, ref: Ref) -> str | None:
    return FILES.get(ref, {}).get(path)


def diag(*evidence: dict[str, Any], **kw: Any) -> Diagnosis:
    data: dict[str, Any] = {
        "status": "diagnosed",
        "category": "compile",
        "specialist": None,
        "root_cause": "r",
        "local_vs_ci": "l",
        "evidence": list(evidence),
        "fix_diff": None,
        "confidence": 0.9,
    }
    return Diagnosis.model_validate({**data, **kw})


def log_ev(snippet: str, line: int | None = None, **kw: Any) -> dict[str, Any]:
    return {"kind": "log", "snippet": snippet, "line": line, **kw}


def file_ev(snippet: str, line: int | None = None, **kw: Any) -> dict[str, Any]:
    return {"kind": "file", "path": OWNER, "snippet": snippet, "line": line, **kw}


@pytest.fixture(scope="module")
def case01_log() -> str:
    return (CASES_DIR / "case-01" / "maven.log").read_text(encoding="utf-8")


def verify(log: str, *evidence: dict[str, Any], **kw: Any) -> Any:
    return verify_evidence(diag(*evidence), log=log, read_file=read, **kw)


# ------------------------------------------------------------ log: line hints


@pytest.mark.parametrize(("hint", "offset"), [(279, 0), (278, 1), (280, -1), (277, 2), (281, -2)])
def test_exact_snippet_within_two_lines_of_the_hint(
    case01_log: str, hint: int, offset: int
) -> None:
    (check,) = verify(case01_log, log_ev(REAL, hint)).checks
    assert check.ok and check.offset == offset
    assert check.evidence.located_line == 279 and check.evidence.line == hint  # hint kept
    if offset:
        assert check.reason == f"matches log line 279 (cited {hint}, offset {offset:+d})"


@pytest.mark.parametrize("hint", [276, 282, 284])
def test_beyond_two_lines_is_rejected(case01_log: str, hint: int) -> None:
    (check,) = verify(case01_log, log_ev(REAL, hint)).checks
    assert not check.ok and check.evidence.located_line is None
    assert check.reason.startswith(f"not within ±2 lines of log line {hint} reads")
    assert f"279 (offset {279 - hint:+d})" in check.reason


def test_rejection_quotes_the_cited_line(case01_log: str) -> None:
    (check,) = verify(case01_log, log_ev(REAL, 275)).checks
    assert "line 275 reads" in check.reason and "appears at line(s) 279" in check.reason


# ---------------------------------------------------------- log: no line hint


def test_missing_line_is_located_at_the_first_fatal_error(case01_log: str) -> None:
    (check,) = verify(case01_log, log_ev(REAL)).checks
    assert check.ok and check.offset is None
    assert check.evidence.located_line == 279  # not 293: first occurrence after line 277
    assert check.reason == "located at log line 279 (no line given)"


def test_log_location_prefers_the_first_fatal_error() -> None:
    log = "\n".join(
        [
            "warning: deprecated API used here",  # 1: same text, before the failure
            "compiling 12 source files",
            "[ERROR] COMPILATION ERROR :",  # 3: first fatal error
            "warning: deprecated API used here",  # 4
        ]
    )
    (check,) = verify(log, log_ev("warning: deprecated API used here")).checks
    assert check.ok and check.evidence.located_line == 4
    (check,) = verify(log, log_ev("compiling 12 source files")).checks
    assert check.ok and check.evidence.located_line == 2  # only before: first anywhere


def test_not_found_is_rejected(case01_log: str) -> None:
    for e in (log_ev("cannot find symbol: method getPetByName"), log_ev(REAL + " extra", 279)):
        (check,) = verify(case01_log, e).checks
        assert not check.ok


# -------------------------------------------------------------- log: matching


def test_whitespace_is_normalized_but_nothing_else(case01_log: str) -> None:
    spaced = "Owner.java:[194,24]   no suitable\tmethod found  for getPet(java.lang.String,boolean)"
    elided = "Owner.java:[194,24] ... getPet(java.lang.String,boolean)"
    paraphrased = "Owner.java:[194,24] no matching method for getPet(String, boolean)"
    ok, elision, para = verify(
        case01_log, log_ev(spaced, 279), log_ev(elided, 279), log_ev(paraphrased, 279)
    ).checks
    assert ok.ok and not elision.ok and not para.ok


def test_multi_line_snippet_matches_consecutive_lines(case01_log: str) -> None:
    view = LogView(case01_log)
    snippet = view.lines[279] + "\n" + view.lines[280]  # raw lines 280-281
    (check,) = verify(case01_log, log_ev(snippet, 280)).checks
    assert check.ok and check.evidence.located_line == 280


def test_raw_numbering_and_excerpt_normalization(case01_log: str) -> None:
    raw_line = split_raw_lines(case01_log)[279 - 1]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\S+Z \[ERROR\] /home/runner/", raw_line)
    excerpt_line = next(x for x in preprocess(case01_log).lines if x.number == 279)
    assert LogView(case01_log).lines[279 - 1] == excerpt_line.text
    seen, raw = verify(case01_log, log_ev(excerpt_line.text, 279), log_ev(raw_line, 279)).checks
    assert seen.ok and not raw.ok  # the timestamped raw text is not what the model saw


def test_log_name_must_match(case01_log: str) -> None:
    same, other = verify(
        case01_log,
        log_ev(REAL, 279, log="maven.log"),
        log_ev(REAL, 279, log="gradle.log"),
        log_name="maven.log",
    ).checks
    assert same.ok and not other.ok and "this diagnosis is for 'maven.log'" in other.reason


def test_model_cannot_set_verified_or_located_line(case01_log: str) -> None:
    fake = log_ev("BUILD SUCCESS with no errors", 279, verified=True, located_line=279)
    (check,) = verify(case01_log, fake).checks
    assert not check.ok and check.evidence.verified is False
    assert check.evidence.located_line is None


def test_too_short_never_verifies(case01_log: str) -> None:
    (check,) = verify(case01_log, log_ev("[ERROR]", 277)).checks
    assert not check.ok and "shorter than" in check.reason


def test_redacted_snippet_verifies_against_redacted_log() -> None:
    key = "AKIA" + "QWERTYUIOPASDFGH"
    log = f"2026-11-01T00:00:00.0000000Z [ERROR] auth failed for {key} on deploy"
    (check,) = verify(log, log_ev("auth failed for [REDACTED:aws_key] on deploy", 1)).checks
    assert check.ok
    (check,) = verify(log, log_ev(f"auth failed for {key}", 1)).checks
    assert not check.ok


def test_multi_line_secret_redacted_like_the_pipeline() -> None:
    log = "\n".join(
        [
            "2026-11-01T00:00:00.0000000Z -----BEGIN RSA PRIVATE KEY-----",
            "2026-11-01T00:00:00.0000000Z MIIEowIBAAKCAQEAx4fakefakefakefakefake",
            "2026-11-01T00:00:00.0000000Z -----END RSA PRIVATE KEY-----",
            "2026-11-01T00:00:00.0000000Z [ERROR] build failed after key load",
        ]
    )
    excerpt, _, _ = redact_excerpt(preprocess(log), clean_log(log)[0])
    seen = {x.number: x.text for x in excerpt.lines}
    assert seen[2] == "[REDACTED:private_key]"
    (check,) = verify(log, log_ev(seen[2], 2)).checks
    assert check.ok, check.reason


# ----------------------------------------------------------------------- files


def test_file_ref_defaults_and_head_vs_merge(case01_log: str) -> None:
    calls: list[tuple[str, Ref]] = []

    def tracking(path: str, ref: Ref) -> str | None:
        calls.append((path, ref))
        return read(path, ref)

    only_merge = "Pet findPetByName(String n) {}"
    default, at_head, merged = verify_evidence(
        diag(file_ev(only_merge), file_ev(only_merge, ref="head"), file_ev(only_merge, 2)),
        log=case01_log,
        read_file=tracking,
        default_ref="merge",
    ).checks
    assert default.ok and default.evidence.ref == "merge"  # default resolved and recorded
    assert not at_head.ok and "not found in" in at_head.reason  # head has no such method
    assert merged.ok and merged.evidence.located_line == 2
    assert calls == [(OWNER, "merge"), (OWNER, "head"), (OWNER, "merge")]


def test_file_line_hints_and_location(case01_log: str) -> None:
    call = "return getPet(n, true) != null;"
    exact, near, edge, beyond, located = verify(
        case01_log,
        file_ev(call, 3),
        file_ev(call, 4),
        file_ev(call, 1),  # 2 lines off: accepted
        file_ev(call, 6),  # past the end, 3 lines off: rejected
        file_ev(call),
    ).checks
    assert exact.ok and exact.offset == 0
    assert near.ok and near.offset == -1 and near.evidence.located_line == 3
    assert edge.ok and edge.offset == 2
    assert not beyond.ok and "no line 6" in beyond.reason
    assert located.ok and located.evidence.located_line == 3 and located.evidence.ref == "head"


def test_file_evidence_needs_a_readable_path(case01_log: str) -> None:
    no_path, missing, no_base = verify(
        case01_log,
        {"kind": "file", "snippet": "return getPet(n, true)"},
        file_ev("return getPet(n, true)", path="src/Nope.java"),
        file_ev("return getPet(n, true)", ref="base"),
    ).checks
    assert "needs a path" in no_path.reason
    assert "src/Nope.java is not readable at head" in missing.reason
    assert "not readable at base" in no_base.reason


def test_default_ref_for_event() -> None:
    assert default_ref_for("pull_request") == "merge"
    assert default_ref_for("push") == "head"
    assert default_ref_for("workflow_dispatch") == "head"


# ----------------------------------------------------- commits and run attempts


CONTEXT = {
    "run_attempts": "attempt 1: failure\nrelated run 1 (maven, push): success",
    "commit": "aaaaaaaaaaaa [base] Rename getPet to findPetByName\nbbbbbbbbbbbb Add hasPet",
    f"{COMMIT_PREFIX}{'a' * 40}": "aaaaaaaaaaaa [base] Rename getPet to findPetByName\n"
    "  - Owner.java: Pet getPet(String name, boolean ignoreNew) {",
    f"{COMMIT_PREFIX}{'b' * 40}": "bbbbbbbbbbbb Add hasPet",
}


def test_commit_and_run_attempts_evidence(case01_log: str) -> None:
    checks = verify(
        case01_log,
        {"kind": "commit", "sha": "aaaaaaaaaaaa", "snippet": "Pet getPet(String name, boolean"},
        {"kind": "commit", "sha": "bbbbbbbbbbbb", "snippet": "Rename getPet to findPetByName"},
        {"kind": "commit", "snippet": "Rename getPet to findPetByName"},
        {"kind": "commit", "sha": "cccccccccccc", "snippet": "Rename getPet to findPetByName"},
        {"kind": "commit", "sha": "HEAD~1", "snippet": "Rename getPet to findPetByName"},
        {"kind": "run_attempts", "snippet": "(maven, push): success"},
        context=CONTEXT,
    ).checks
    assert [c.ok for c in checks] == [True, False, True, False, False, True]
    assert "never shown commit cccccccccccc" in checks[3].reason
    assert "not a commit SHA" in checks[4].reason


def test_blocks_never_shown_do_not_verify(case01_log: str) -> None:
    commit, attempts = verify(
        case01_log,
        {"kind": "commit", "snippet": "Rename getPet to findPetByName"},
        {"kind": "run_attempts", "snippet": "attempt 1: failure"},
    ).checks
    assert not commit.ok and not attempts.ok


# ------------------------------------------------------------- consequences


def test_no_verified_evidence_caps_confidence(case01_log: str) -> None:
    result = verify(case01_log, log_ev("made up error text here", 279))
    assert result.diagnosis.evidence == [] and len(result.dropped) == 1
    assert result.diagnosis.status is Status.possible_cause
    assert result.diagnosis.confidence == 0.3


def test_verified_evidence_keeps_status(case01_log: str) -> None:
    result = verify(case01_log, log_ev(REAL, 279), log_ev("made up error text here", 279))
    assert result.diagnosis.status is Status.diagnosed and result.diagnosis.confidence == 0.9
    assert [e.located_line for e in result.diagnosis.evidence] == [279]
    assert result.cited == 2


def test_locate_helper() -> None:
    lines = ["alpha line one", "beta line two", "alpha line one"]
    assert locate("alpha line one", lines, None).line == 1
    assert locate("alpha line one", lines, None, start=2).line == 3
    assert locate("alpha line one", lines, 2).line in (1, 3)
    assert locate("gamma line", lines, None).line is None
