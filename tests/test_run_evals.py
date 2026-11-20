"""Eval runner: scoring, aggregation, error handling, table and results file."""

import json
from pathlib import Path
from typing import Any

import pytest

from agent.base import AgentRequest, AgentResult, AgentUsage
from agent.errors import ModelUnavailableError
from agent.schema import Category, Diagnosis, Evidence, Status
from agent.stub import StubAgent
from evals import run_evals
from evals.loader import list_cases
from evals.run_evals import (
    RunScore,
    _covers,
    aggregate,
    per_log,
    render_table,
    results_json,
    run_eval,
    targets,
)
from guardrails.budget import Budget, BudgetExceeded, BudgetLimits

REAL = "error: package com.google.common.base does not exist"  # case-02 gradle.log:239


class DependencyAgent:
    """Answers case-02 correctly with one real and one fabricated piece of evidence."""

    name = "fake-dependency"

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        d = Diagnosis(
            status=Status.diagnosed,
            category=Category.dependency,
            specialist="build",
            root_cause="Guava is declared in pom.xml but not in build.gradle.",
            local_vs_ci="The developer built with Maven only.",
            evidence=[
                Evidence(kind="log", line=239, snippet=REAL),
                Evidence(kind="log", line=239, snippet="error: cannot resolve guava 33.0"),
            ],
            fix_diff=None,
            confidence=0.9,
        )
        usage = AgentUsage(
            model_id="fake@1", llm_calls=2, input_tokens=1000, output_tokens=100, cost_usd=0.01
        )
        return AgentResult(diagnosis=d, usage=usage, tool_calls=3, structured_output_fallback=True)


class FailingAgent:
    name = "failing"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        raise self.exc


def run(agent: Any, pairs: list[tuple[str, str]], n: int = 1) -> list[RunScore]:
    return run_eval(pairs, agent=agent, repo=None, n=n, limits=BudgetLimits())


# ------------------------------------------------------------------ scoring


def test_targets_cover_every_failed_log() -> None:
    assert targets(None) == [(c, log) for c in list_cases() for log in run_evals.case_logs(c)]
    assert targets(["case-02"]) == [("case-02", "gradle.log")]


def test_scores_a_correct_answer_with_one_fabricated_item() -> None:
    (s,) = run(DependencyAgent(), [("case-02", "gradle.log")])
    assert s.error is None
    assert s.category == "dependency" and s.category_ok is True
    assert s.status_ok is (s.status == "diagnosed")
    assert (s.evidence_cited, s.evidence_verified, s.evidence_hallucinated) == (2, 1, 1)
    assert s.has_verified_evidence
    assert s.sources_expected == ("log", "pom.xml", "build.gradle")
    assert s.sources_covered == ("log",)  # file evidence cannot verify without a repo
    assert s.agent_status == "diagnosed" and s.agent_confidence == 0.9
    assert s.structured_output_fallback and s.tool_calls == 3
    assert (s.llm_calls, s.input_tokens, s.output_tokens, s.cost_usd) == (2, 1000, 100, 0.01)
    assert s.model_id == "fake@1"
    assert s.confidence_in_range is None  # case-02 has no expected range


def test_wrong_category_and_stub_scores() -> None:
    scores = run(StubAgent(), [("case-01", "maven.log"), ("case-03", "gradle.log")])
    assert all(s.category == "unknown" and s.category_ok is False for s in scores)
    assert all(s.status == "possible_cause" and s.status_ok is False for s in scores)
    assert all(s.evidence_verified == 1 for s in scores)  # the stub cites the first error


def test_n_runs_per_log() -> None:
    scores = run(StubAgent(), [("case-02", "gradle.log")], n=3)
    assert [s.run for s in scores] == [1, 2, 3]


@pytest.mark.parametrize(
    "exc",
    [BudgetExceeded("wall_clock", 120, 121), ModelUnavailableError("model is down")],
)
def test_budget_and_model_errors_are_recorded_not_raised(exc: Exception) -> None:
    (s,) = run(FailingAgent(exc), [("case-02", "gradle.log")])
    assert s.error is not None and type(exc).__name__ in s.error
    assert s.category is None and s.evidence_cited == 0
    total = aggregate([s])
    assert total.errors == 1 and total.category_accuracy is None


def test_other_exceptions_are_bugs_and_propagate() -> None:
    with pytest.raises(RuntimeError):
        run(FailingAgent(RuntimeError("bug")), [("case-02", "gradle.log")])


@pytest.mark.parametrize(
    ("expected", "labels", "covered"),
    [
        ("log", ["log"], True),
        ("commit", ["log"], False),
        ("pom.xml", ["pom.xml"], True),
        ("build.gradle", ["pom.xml"], False),
        ("src/main/java/a/Owner.java", ["src/main/java/a/Owner.java"], True),
        ("Owner.java", ["src/main/java/a/Owner.java"], True),
        ("log", ["src/log"], False),  # a file is never the "log" source
        ("pom.xml", ["xpom.xml"], False),
    ],
)
def test_source_coverage(expected: str, labels: list[str], covered: bool) -> None:
    assert _covers(expected, labels) is covered


# -------------------------------------------------------------- aggregation


def _score(**kw: Any) -> RunScore:
    base: dict[str, Any] = {
        "case": "c",
        "log": "l",
        "run": 1,
        "category_ok": True,
        "status_ok": True,
        "confidence": 0.8,
        "cost_usd": 0.0,
    }
    base.update(kw)
    return RunScore(**base)


def test_aggregate_rates() -> None:
    scores = [
        _score(
            evidence_cited=4,
            evidence_verified=3,
            sources_expected=("log", "pom.xml"),
            sources_covered=("log",),
            confidence_in_range=True,
            fix_proposed=True,
            fix_applies=True,
            cost_usd=0.5,
            latency_s=10.0,
        ),
        _score(
            run=2,
            category_ok=False,
            evidence_cited=2,
            evidence_verified=0,
            sources_expected=("log", "pom.xml"),
            confidence_in_range=False,
            fix_proposed=True,
            cost_usd=0.25,
            latency_s=20.0,
        ),
        _score(run=3, error="ModelUnavailableError: down"),
    ]
    a = aggregate(scores)
    assert (a.runs, a.errors) == (3, 1)
    assert a.category_accuracy == 0.5 and a.status_accuracy == 1.0
    assert a.runs_with_verified_evidence == 0.5
    assert (a.evidence_cited, a.evidence_verified) == (6, 3)
    assert a.verified_evidence_rate == 0.5 and a.hallucinated_evidence_rate == 0.5
    assert a.sources_coverage == 0.25
    assert a.confidence_in_range == 0.5
    assert a.fix_applies_rate == 0.5
    assert a.cost_usd_total == 0.75
    assert a.latency_s is not None and a.latency_s.mean == 15.0 and a.latency_s.std == 5.0


def test_unknown_price_makes_total_cost_unknown() -> None:
    assert aggregate([_score(cost_usd=0.1), _score(run=2, cost_usd=None)]).cost_usd_total is None


def test_no_evidence_means_rates_are_undefined_not_zero() -> None:
    a = aggregate([_score()])
    assert a.verified_evidence_rate is None and a.hallucinated_evidence_rate is None


def test_per_log_groups_in_order() -> None:
    rows = per_log(
        [_score(case="a", log="x"), _score(case="b", log="y"), _score(case="a", log="x", run=2)]
    )
    assert [(r.group["case"], r.group["log"], r.runs) for r in rows] == [
        ("a", "x", 2),
        ("b", "y", 1),
    ]


# ------------------------------------------------------------ table + file


def test_table_has_a_row_per_log_and_a_total() -> None:
    scores = run(DependencyAgent(), [("case-02", "gradle.log")], n=2)
    table = render_table(per_log(scores), aggregate(scores))
    lines = table.splitlines()
    assert lines[0].startswith("case/log") and "halluc." in lines[0]
    assert lines[2].startswith("case-02/gradle.log") and "50%" in lines[2]
    assert lines[-1].startswith("ALL") and "$0.0200" in lines[-1]


def test_results_json_records_provenance() -> None:
    scores = run(DependencyAgent(), [("case-02", "gradle.log")])
    meta = run_evals.provenance(
        agent="fake-dependency", n=1, limits=BudgetLimits(), repo_reader=False, scores=scores
    )
    data = results_json(meta, scores, per_log(scores), aggregate(scores))
    assert set(data) == {"meta", "summary", "per_log", "runs"}
    assert meta["model_ids"] == ["fake@1"] and meta["n"] == 1
    assert "generalist.md" in meta["prompt_hashes"]
    assert len(meta["prompt_hashes"]["generalist.md"]) == 64
    assert meta["git_sha"] and meta["budget_limits"]["max_input_tokens"] == 60_000
    assert data["runs"][0]["evidence_hallucinated"] == 1
    json.dumps(data)  # serializable


def test_main_dry_run_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.json"
    assert run_evals.main(["--dry-run", "--n", "1", "--case", "case-02", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "case-02/gradle.log" in printed and "ALL" in printed
    data = json.loads(out.read_text())
    assert data["meta"]["agent"] == "stub" and data["summary"]["runs"] == 1


def test_main_rejects_bad_n() -> None:
    with pytest.raises(SystemExit):
        run_evals.main(["--dry-run", "--n", "0"])
