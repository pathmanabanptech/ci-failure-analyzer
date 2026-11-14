"""Dry-run pipeline: stage order, spans, redaction before the agent, no model call."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import cli
import telemetry
from adapters.base import Report, RunDetails, RunRef
from adapters.logs.local import LocalLogSource
from agent.base import AgentRequest, AgentResult
from agent.schema import Category, Diagnosis, Evidence, Status
from agent.stub import StubAgent
from evals.loader import CASES_DIR, case_logs, list_cases
from guardrails.budget import Budget, BudgetExceeded
from guardrails.gating import AlertLevel
from pipeline import PipelineResult, analyze

STAGES = ["load", "preprocess", "redact", "agent", "verify", "gate", "sanitize", "notify"]
FAKE_AWS_KEY = "AKIAQWERTYUIOPASDFGH"


@dataclass
class RecordingNotifier:
    reports: list[Report] = field(default_factory=list)

    def notify(self, report: Report) -> None:
        self.reports.append(report)


@dataclass
class RecordingAgent:
    """Wraps another agent and records what it was given."""

    inner: StubAgent = field(default_factory=StubAgent)
    requests: list[AgentRequest] = field(default_factory=list)
    name: str = "recording"

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        self.requests.append(request)
        return self.inner.diagnose(request, budget)


class SecretInjectingSource(LocalLogSource):
    """A real case log with a fake AWS key appended to line ``line``."""

    def __init__(self, line: int) -> None:
        super().__init__(CASES_DIR)
        self._line = line

    def get_log(self, run: RunRef) -> str:
        lines = super().get_log(run).split("\n")
        lines[self._line - 1] += f" aws key {FAKE_AWS_KEY}"
        return "\n".join(lines)


def run(
    case: str,
    log: str,
    *,
    source: LocalLogSource | None = None,
    agent: object | None = None,
) -> tuple[PipelineResult, list[ReadableSpan], RecordingNotifier]:
    source = source or LocalLogSource(CASES_DIR)
    exporter = InMemorySpanExporter()
    provider = telemetry.make_provider(exporter)
    notifier = RecordingNotifier()
    result = analyze(
        source.ref_for(case, log),
        logs=source,
        repo=None,
        notifier=notifier,
        agent=agent or StubAgent(),  # type: ignore[arg-type]
        tracer=telemetry.get_tracer(provider),
    )
    return result, list(exporter.get_finished_spans()), notifier


ALL_LOGS = [(c, log) for c in list_cases() for log in case_logs(c)]


@pytest.mark.parametrize(("case", "log"), ALL_LOGS)
def test_every_case_log_runs_end_to_end(case: str, log: str) -> None:
    result, _spans, notifier = run(case, log)
    Diagnosis.model_validate_json(result.diagnosis.model_dump_json())
    # The stub cites the first fatal error; the real verifier must accept it.
    assert result.dropped_evidence == 0
    assert [e.verified for e in result.diagnosis.evidence] == [True]
    assert result.usage.llm_calls == 0 and result.usage.model_id is None
    assert result.decision.level is AlertLevel.possible_cause
    assert len(notifier.reports) == 1 and notifier.reports[0].diagnosis == result.diagnosis


def test_one_span_per_stage_in_order_under_one_root() -> None:
    _, spans, _ = run("case-01", "maven.log")
    root = next(s for s in spans if s.name == "analyze")
    children = sorted((s for s in spans if s.parent is not None), key=lambda s: s.start_time or 0)
    assert [s.name for s in children] == STAGES
    assert all(s.parent and s.parent.span_id == root.context.span_id for s in children)
    assert len({s.context.trace_id for s in spans}) == 1

    attrs = {s.name: dict(s.attributes or {}) for s in spans}
    assert attrs["preprocess"]["first_error.line"] == 277
    assert attrs["agent"]["llm_calls"] == 0
    assert attrs["verify"]["evidence.verified"] == 1
    assert attrs["notify"]["notified"] is True


def test_case03_gradle_spans_report_oom_and_cancel_separately() -> None:
    _, spans, _ = run("case-03", "gradle.log")
    pre = next(dict(s.attributes or {}) for s in spans if s.name == "preprocess")
    assert pre["first_error.kind"] == "out_of_memory"
    assert pre["termination.kind"] == "cancelled"


def test_spans_never_carry_log_text() -> None:
    _, spans, _ = run("case-01", "maven.log", source=SecretInjectingSource(277))
    for s in spans:
        for value in (s.attributes or {}).values():
            assert FAKE_AWS_KEY not in str(value)
            assert "COMPILATION ERROR" not in str(value)


def test_agent_only_sees_redacted_excerpt() -> None:
    agent = RecordingAgent()
    result, _spans, notifier = run(
        "case-01", "maven.log", source=SecretInjectingSource(277), agent=agent
    )
    (request,) = agent.requests
    seen = "\n".join(line.text for line in request.excerpt.lines)
    assert FAKE_AWS_KEY not in seen and "[REDACTED:aws_key]" in seen
    assert request.excerpt.first_error is not None
    assert FAKE_AWS_KEY not in request.excerpt.first_error.line.text
    # Line numbers survive redaction; the redacted line still verifies.
    assert request.excerpt.first_error.line.number == 277
    assert result.diagnosis.evidence[0].verified
    assert result.redactions == {"aws_key": 1}
    assert FAKE_AWS_KEY not in result.diagnosis.model_dump_json()
    assert FAKE_AWS_KEY not in notifier.reports[0].diagnosis.model_dump_json()


@dataclass
class FabricatingAgent:
    name: str = "fabricating"

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        return AgentResult(
            Diagnosis(
                status=Status.diagnosed,
                category=Category.compile,
                specialist="build",
                root_cause="made up",
                local_vs_ci="made up",
                evidence=[Evidence(source="log", line=10, snippet="error: this is not there")],
                fix_diff=None,
                confidence=0.95,
                auto_fixable=True,
            )
        )


def test_unverified_evidence_is_capped_and_nothing_is_posted() -> None:
    result, _spans, notifier = run("case-01", "maven.log", agent=FabricatingAgent())
    assert result.dropped_evidence == 1
    assert result.diagnosis.status is Status.possible_cause
    assert result.diagnosis.confidence == 0.3
    assert result.diagnosis.auto_fixable is False
    assert result.decision.level is AlertLevel.none and not result.notified
    assert notifier.reports == []


@dataclass
class OverBudgetAgent:
    name: str = "over-budget"

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        budget.charge_input(10**9)
        raise AssertionError("unreachable")


def test_budget_exceeded_propagates_and_is_recorded_on_spans() -> None:
    exporter = InMemorySpanExporter()
    tracer = telemetry.get_tracer(telemetry.make_provider(exporter))
    source = LocalLogSource(CASES_DIR)
    notifier = RecordingNotifier()
    with pytest.raises(BudgetExceeded):
        analyze(
            source.ref_for("case-01", "maven.log"),
            logs=source,
            repo=None,
            notifier=notifier,
            agent=OverBudgetAgent(),
            tracer=tracer,
        )
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "verify" not in spans and notifier.reports == []
    assert spans["agent"].status.status_code.name == "ERROR"


def test_run_attempts_context_verifies() -> None:
    from pipeline import run_attempts_text

    source = LocalLogSource(CASES_DIR)
    details: RunDetails = source.get_run(source.ref_for("case-01", "maven.log"))
    assert "attempt 1: failure" in run_attempts_text(details)


# --------------------------------------------------------------------- CLI


def test_cli_analyze_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", "--case", "case-01", "--dry-run"]) == 0
    out, err = capsys.readouterr()
    d = Diagnosis.model_validate(json.loads(out))  # stdout is exactly one Diagnosis JSON
    assert d.build_tool == "maven"
    for stage in [*STAGES, "analyze"]:
        assert re.search(rf"^\[span\] {stage} ", err, re.M), stage
    assert "llm_calls=0" in err


def test_cli_analyze_picks_log(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", "--case", "case-03", "--log", "gradle.log", "--dry-run"]) == 0
    assert Diagnosis.model_validate_json(capsys.readouterr().out).build_tool == "gradle"


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--case", "case-01"],  # no LLM agent yet
        ["analyze", "--case", "case-01", "--log", "nope.log", "--dry-run"],
        ["analyze", "--case", "case-99", "--dry-run"],
    ],
)
def test_cli_analyze_errors(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error:")


def test_no_model_sdk_imported() -> None:
    for mod in ("strands", "anthropic", "boto3", "ollama"):
        assert mod not in sys.modules
