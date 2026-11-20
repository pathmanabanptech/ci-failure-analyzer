"""Dry-run pipeline: stage order, spans, redaction before the agent, no model call."""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import cli
import telemetry
from adapters.base import NotFoundError, PatchCheck, Report, RunDetails, RunRef
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
ROOT = Path(__file__).resolve().parent.parent


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
    assert result.dropped_evidence == ()
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
                evidence=[Evidence(kind="log", line=10, snippet="error: this is not there")],
                fix_diff=None,
                confidence=0.95,
                auto_fixable=True,
            )
        )


def test_unverified_evidence_is_capped_and_nothing_is_posted() -> None:
    result, _spans, notifier = run("case-01", "maven.log", agent=FabricatingAgent())
    assert len(result.dropped_evidence) == 1
    assert result.diagnosis.status is Status.possible_cause
    assert result.diagnosis.confidence == 0.3
    assert result.diagnosis.auto_fixable is False
    assert result.decision.level is AlertLevel.none and not result.notified
    assert notifier.reports == []


OWNER = "src/main/java/org/springframework/samples/petclinic/owner/Owner.java"
OWNER_SRC = (
    "class Owner {\n"
    "    boolean hasPet(String n) {\n"
    "        return getPet(n, true) != null;\n"
    "    }\n"
    "}\n"
)


@dataclass
class OwnerRepo:
    def read_file(self, sha: str, path: str) -> str:
        if path != OWNER:
            raise NotFoundError(path)
        return OWNER_SRC

    def recent_commits(self, sha: str, limit: int = 10, path: str | None = None) -> list[Any]:
        return []

    def read_file_at_merge(self, base_sha: str, head_sha: str, path: str) -> str:
        return self.read_file(head_sha, path)

    def symbol_history(
        self, base_sha: str | None, head_sha: str, identifier: str, path: str | None = None
    ) -> list[Any]:
        return []

    def check_patch(self, patch: str, head_sha: str, base_sha: str | None = None) -> PatchCheck:
        # Stand-in for git apply: the patch applies if every removed line is in the file.
        removed = [
            ln[1:] for ln in patch.splitlines() if ln.startswith("-") and not ln.startswith("---")
        ]
        ok = all(r in OWNER_SRC for r in removed) and f"--- a/{OWNER}" in patch
        return PatchCheck(ok, "applies to merge" if ok else "does not apply to merge")


@dataclass
class FixingAgent:
    fix: str
    name: str = "fixing"

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        result = StubAgent().diagnose(request, budget)
        return replace(result, diagnosis=result.diagnosis.model_copy(update={"fix_diff": self.fix}))


def _diff(after: str) -> str:
    return "".join(
        difflib.unified_diff(
            OWNER_SRC.splitlines(True), after.splitlines(True), f"a/{OWNER}", f"b/{OWNER}"
        )
    )


# Plausible-looking but invented context: no such line in Owner.java.
FABRICATED_FIX = (
    f"--- a/{OWNER}\n+++ b/{OWNER}\n@@ -2,3 +2,3 @@\n"
    "     boolean hasPet(String n) {\n"
    "-        return pets.get(n, false) != null;\n"
    "+        return pets.get(n) != null;\n"
    "     }\n"
)


@pytest.mark.parametrize(
    ("fix", "kept"),
    [
        (_diff(OWNER_SRC.replace("getPet(n, true)", "findPetByName(n, true)")), True),
        (FABRICATED_FIX, False),
    ],
)
def test_fix_diff_kept_only_if_it_applies_at_head(fix: str, kept: bool) -> None:
    source = LocalLogSource(CASES_DIR)
    tracer = telemetry.get_tracer(telemetry.make_provider(InMemorySpanExporter()))
    result = analyze(
        source.ref_for("case-01", "maven.log"),
        logs=source,
        repo=OwnerRepo(),
        notifier=RecordingNotifier(),
        agent=FixingAgent(fix),
        tracer=tracer,
    )
    assert result.fix_check is not None and result.fix_check.ok is kept
    assert (result.diagnosis.fix_diff is not None) is kept
    assert result.agent_output.diagnosis.fix_diff == fix  # raw output kept for --debug


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


def test_cli_analyze_runs_every_failed_log(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", "--case", "case-01", "--dry-run"]) == 0
    out, err = capsys.readouterr()
    by_log = json.loads(out)  # stdout: one Diagnosis per log, keyed by log name
    assert list(by_log) == ["maven.log", "gradle.log"]
    assert {log: Diagnosis.model_validate(d).build_tool for log, d in by_log.items()} == {
        "maven.log": "maven",
        "gradle.log": "gradle",
    }
    assert "# case-01/maven.log" in err and "# case-01/gradle.log" in err
    for stage in [*STAGES, "analyze"]:
        assert len(re.findall(rf"^\[span\] {stage} ", err, re.M)) == 2, stage
    assert "llm_calls=0" in err


def test_cli_analyze_skips_successful_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", "--case", "case-02", "--dry-run"]) == 0  # maven passed
    assert list(json.loads(capsys.readouterr().out)) == ["gradle.log"]


def test_cli_analyze_single_log(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze", "--case", "case-03", "--log", "gradle.log", "--dry-run"]) == 0
    by_log = json.loads(capsys.readouterr().out)
    assert list(by_log) == ["gradle.log"]
    assert Diagnosis.model_validate(by_log["gradle.log"]).build_tool == "gradle"


def test_cli_analyze_debug(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["analyze", "--case", "case-01", "--log", "maven.log", "--dry-run", "--debug"]
    assert cli.main(argv) == 0
    out, err = capsys.readouterr()
    assert list(json.loads(out)) == ["maven.log"]  # debug output never goes to stdout
    assert "--- debug: maven.log ---" in err
    assert "raw model output (before guardrails):" in err and '"confidence": 0.5' in err
    assert (
        "[accepted] 'maven.log:277' (line hint 277, located_line 277): matches log line 277" in err
    )
    assert "snippet: '[ERROR] COMPILATION ERROR :'" in err
    assert "fix_diff: none proposed" in err
    assert "tool calls (0):\n  (none)" in err
    assert "category evidence requirements: NOT met, missing a known category" in err
    assert "after guardrails: possible_cause / unknown" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--case", "case-01", "--log", "nope.log", "--dry-run"],
        ["analyze", "--case", "case-99", "--dry-run"],
    ],
)
def test_cli_analyze_errors(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("error:")


def test_cli_analyze_without_model_config_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    assert cli.main(["analyze", "--case", "case-01"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "error: MODEL_PROVIDER is not set" in err and "Traceback" not in err


def test_dry_run_never_loads_a_model_sdk() -> None:
    """Fresh interpreter: other tests in this process may have imported model SDKs."""
    code = (
        "import sys, cli\n"
        "rc = cli.main(['analyze', '--case', 'case-01', '--dry-run'])\n"
        "loaded = [m for m in ('strands', 'anthropic', 'boto3', 'ollama') if m in sys.modules]\n"
        "print('LOADED', loaded, file=sys.stderr)\n"
        "sys.exit(rc)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "LOADED []" in proc.stderr


def test_compact_spans_stay_on_one_line() -> None:
    exporter = InMemorySpanExporter()
    tracer = telemetry.get_tracer(telemetry.make_provider(exporter))
    with tracer.start_as_current_span("tool") as span:
        span.set_attribute("description", "Read a file.\n\n    Args:\n        path: x")
    (s,) = exporter.get_finished_spans()
    line = telemetry.compact_line(s)
    assert line.count("\n") == 1 and "description=Read a file. Args: path: x" in line


def test_load_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "MODEL_PROVIDER=ollama\n"
        "export TRIAGE_MODEL_ID='qwen2.5:7b-instruct@845dbda0ea48'\n"
        'OLLAMA_HOST="http://localhost:11434"  \n'
        "MODEL_TIMEOUT_S=120  # seconds\n"
        "BUDGET_TIMEOUT_S=300\n"
        "not a setting\n"
        "\n"
    )
    environ = {"BUDGET_TIMEOUT_S": "600"}  # already set in the shell: wins
    loaded = cli.load_env_file(env_file, environ)
    assert environ == {
        "BUDGET_TIMEOUT_S": "600",
        "MODEL_PROVIDER": "ollama",
        "TRIAGE_MODEL_ID": "qwen2.5:7b-instruct@845dbda0ea48",
        "OLLAMA_HOST": "http://localhost:11434",
        "MODEL_TIMEOUT_S": "120",
    }
    assert "BUDGET_TIMEOUT_S" not in loaded
    assert cli.load_env_file(tmp_path / "missing.env", environ) == []


def test_cli_script_loads_dotenv_but_main_does_not(tmp_path: Path) -> None:
    """Only `python cli.py` reads .env; tests calling cli.main() never do."""
    src = (ROOT / "cli.py").read_text(encoding="utf-8")
    main_body = src[src.index("def main(") : src.index('if __name__ == "__main__":')]
    assert "load_env_file" not in main_body
    assert "load_env_file(ENV_FILE, os.environ)" in src[src.index('if __name__ == "__main__":') :]
