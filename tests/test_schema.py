import pytest
from pydantic import ValidationError

from agent.schema import Category, Diagnosis, Evidence, Status


def _diag(**overrides: object) -> Diagnosis:
    data: dict[str, object] = {
        "status": "diagnosed",
        "category": "compile",
        "specialist": "build",
        "root_cause": "x",
        "local_vs_ci": "y",
        "evidence": [{"kind": "log", "line": 3, "snippet": "error: cannot find symbol"}],
        "fix_diff": None,
        "confidence": 0.9,
    }
    data.update(overrides)
    return Diagnosis.model_validate(data)


def test_diagnosis_round_trips_json() -> None:
    d = _diag()
    assert Diagnosis.model_validate_json(d.model_dump_json()) == d
    assert d.category is Category.compile and d.status is Status.diagnosed


def test_format_category_round_trips() -> None:
    d = _diag(category="format")
    assert d.category is Category("format")
    assert d.category.value == "format"
    assert '"category":"format"' in d.model_dump_json()


def test_defaults_for_code_owned_fields() -> None:
    d = _diag()
    assert d.auto_fixable is False
    assert d.evidence[0].verified is False
    assert d.reroute is None and d.build_tool is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "segfault"},
        {"status": "probably"},
        {"specialist": "docs"},
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"build_tool": "bazel"},
    ],
)
def test_rejects_values_outside_contract(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _diag(**overrides)


def test_evidence_requires_kind_and_snippet() -> None:
    with pytest.raises(ValidationError):
        Evidence.model_validate({"kind": "log"})
    with pytest.raises(ValidationError):
        Evidence.model_validate({"snippet": "error: cannot find symbol"})
    with pytest.raises(ValidationError):
        Evidence.model_validate({"kind": "memory", "snippet": "seen this before"})


def test_evidence_locators_and_code_owned_fields() -> None:
    e = Evidence.model_validate(
        {"kind": "file", "path": "pom.xml", "ref": "merge", "line": 40, "snippet": "<x/>x"}
    )
    assert (e.kind, e.path, e.ref, e.line) == ("file", "pom.xml", "merge", 40)
    assert e.located_line is None and e.verified is False
    with pytest.raises(ValidationError):
        Evidence.model_validate({"kind": "file", "ref": "tip", "snippet": "x"})


@pytest.mark.parametrize(
    ("data", "where"),
    [
        ({"kind": "log", "log": "maven.log", "line": 279}, "maven.log:279"),
        ({"kind": "log", "line": 279, "located_line": 280}, "log:280"),
        ({"kind": "file", "path": "pom.xml", "ref": "merge", "line": 4}, "pom.xml@merge:4"),
        ({"kind": "commit", "sha": "a" * 40}, "commit aaaaaaaaaaaa"),
        ({"kind": "run_attempts"}, "run_attempts"),
    ],
)
def test_evidence_where(data: dict[str, object], where: str) -> None:
    assert Evidence.model_validate({**data, "snippet": "something"}).where() == where
