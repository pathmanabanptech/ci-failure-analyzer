import re
from pathlib import Path
from typing import Any

import pytest

from agent.schema import Category, Status
from evals import loader
from evals.loader import (
    AnswerKeyAccessError,
    CaseError,
    case_logs,
    list_cases,
    load_agent_input,
    load_answer_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANSWER_FIELDS = {
    "category",
    "specialist",
    "status",
    "build_tool",
    "evidence_sources",
    "key_facts",
    "fix_should_build",
    "expected_confidence",
    "dedupe",
    "tags",
    "by_log",
}
BRANCH_VALUE = re.compile(r"(^|[^\w])(case/|refs/heads/|develop\b|feature/)")


def _all_keys_and_strings(node: Any) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    strings: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            keys.add(str(k))
            k2, s2 = _all_keys_and_strings(v)
            keys |= k2
            strings += s2
    elif isinstance(node, list):
        for v in node:
            k2, s2 = _all_keys_and_strings(v)
            keys |= k2
            strings += s2
    elif isinstance(node, str):
        strings.append(node)
    return keys, strings


# ------------------------------------------------------------ real Phase 0 cases


def test_all_three_cases_load() -> None:
    assert list_cases() == ["case-01", "case-02", "case-03"]
    expected_logs = {
        "case-01": ["maven.log", "gradle.log"],
        "case-02": ["gradle.log"],
        "case-03": ["maven.log", "gradle.log"],
    }
    for case_id, logs in expected_logs.items():
        assert case_logs(case_id) == logs
        for log in logs:
            inp = load_agent_input(case_id, log)
            key = load_answer_key(case_id, log)
            assert inp.meta.head_sha and key.key_facts and key.evidence_sources


def test_case_categories() -> None:
    assert load_answer_key("case-01", "maven.log").category is Category.compile
    assert load_answer_key("case-02", "gradle.log").category is Category.dependency
    assert load_answer_key("case-03", "maven.log").category is Category.memory


def test_case03_gradle_resolves_by_log_override() -> None:
    gradle = load_answer_key("case-03", "gradle.log")
    assert gradle.status is Status.possible_cause
    assert gradle.evidence_sources == ["log", "build.gradle"]
    assert gradle.expected_confidence == (0.4, 0.7)
    assert any("cancelled" in f for f in gradle.key_facts)
    # Non-overridden fields fall back to the top-level defaults.
    assert gradle.category is Category.memory and gradle.specialist == "infra"


def test_case03_maven_uses_defaults() -> None:
    maven = load_answer_key("case-03", "maven.log")
    assert maven.status is Status.diagnosed
    assert maven.evidence_sources == ["log", "pom.xml"]
    assert maven.expected_confidence is None


def test_repo_is_normalized() -> None:
    # case-01 meta.yaml has a trailing slash on repo.
    assert load_agent_input("case-01", "maven.log").meta.repo == (
        "pathmanabanptech/petclinic-ci-failures"
    )


@pytest.mark.parametrize(
    ("case_id", "log"),
    [(c, log) for c in ["case-01", "case-02", "case-03"] for log in case_logs(c)],
)
def test_agent_input_has_no_branch_names_or_answer_fields(case_id: str, log: str) -> None:
    dumped = load_agent_input(case_id, log).model_dump(mode="json")
    keys, strings = _all_keys_and_strings(dumped)

    assert not keys & ANSWER_FIELDS
    assert not keys & loader.FORBIDDEN_META_KEYS
    assert not [s for s in strings if BRANCH_VALUE.search(s)]

    key = load_answer_key(case_id, log)
    blob = load_agent_input(case_id, log).model_dump_json()
    for fact in key.key_facts:
        assert fact not in blob


def test_unknown_log_rejected() -> None:
    with pytest.raises(CaseError):
        load_agent_input("case-02", "maven.log")  # Maven passed: no log


@pytest.mark.parametrize("bad", ["../cases", "", "..", "case-01/../case-02"])
def test_case_id_path_traversal_rejected(bad: str) -> None:
    with pytest.raises(CaseError):
        load_agent_input(bad, "maven.log")


# ------------------------------------------------------- answer-key access guard


def test_answer_key_refused_outside_evals(tmp_path: Path) -> None:
    fake_module = tmp_path / "agent_code.py"
    code = compile("load_answer_key('case-01', 'maven.log')", str(fake_module), "exec")
    with pytest.raises(AnswerKeyAccessError):
        exec(code, {"load_answer_key": load_answer_key})


def test_core_packages_never_touch_answer_keys() -> None:
    for pkg in ("agent", "guardrails", "adapters"):
        for py in (REPO_ROOT / pkg).rglob("*.py"):
            src = py.read_text()
            assert "expected.yaml" not in src, py
            assert not re.search(r"^\s*(from|import)\s+evals\b", src, re.M), py


# ------------------------------------------------------ synthetic invalid cases

SHA = "a" * 40


def _write_case(root: Path, meta: str, expected: str = "", logs: tuple[str, ...] = ()) -> str:
    d = root / "case-x"
    d.mkdir()
    (d / "meta.yaml").write_text(meta)
    if expected:
        (d / "expected.yaml").write_text(expected)
    for log in logs:
        (d / log).write_text("log\n")
    return "case-x"


GOOD_META = f"""
repo: o/r
event: push
head_sha: {SHA}
runner_os: ubuntu-latest
jdk: "17"
runs:
  maven: {{ run_id: 1, conclusion: failure, attempts: 1, log: maven.log }}
"""


@pytest.mark.parametrize("leak", ["branch: case/01", "head_ref: case/01", "pr_title: fix oom"])
def test_meta_with_label_leak_rejected(tmp_path: Path, leak: str) -> None:
    cid = _write_case(tmp_path, GOOD_META + leak + "\n", logs=("maven.log",))
    with pytest.raises(CaseError, match="label-leaking"):
        load_agent_input(cid, "maven.log", cases_dir=tmp_path)


def test_meta_with_unknown_key_rejected(tmp_path: Path) -> None:
    cid = _write_case(tmp_path, GOOD_META + "notes: hi\n", logs=("maven.log",))
    with pytest.raises(CaseError):
        load_agent_input(cid, "maven.log", cases_dir=tmp_path)


def test_meta_missing_log_file_rejected(tmp_path: Path) -> None:
    cid = _write_case(tmp_path, GOOD_META)
    with pytest.raises(CaseError, match="missing file"):
        load_agent_input(cid, "maven.log", cases_dir=tmp_path)


def test_by_log_for_unknown_log_rejected(tmp_path: Path) -> None:
    expected = """
category: memory
specialist: infra
status: diagnosed
evidence_sources: [log]
key_facts: [x]
by_log:
  gradle.log: { status: possible_cause }
"""
    cid = _write_case(tmp_path, GOOD_META, expected, logs=("maven.log",))
    with pytest.raises(CaseError, match="unknown logs"):
        load_answer_key(cid, "maven.log", cases_dir=tmp_path)


def test_expected_with_invalid_category_rejected(tmp_path: Path) -> None:
    expected = "category: nope\nspecialist: build\nstatus: diagnosed\n"
    expected += "evidence_sources: [log]\nkey_facts: [x]\n"
    cid = _write_case(tmp_path, GOOD_META, expected, logs=("maven.log",))
    with pytest.raises(CaseError):
        load_answer_key(cid, "maven.log", cases_dir=tmp_path)
