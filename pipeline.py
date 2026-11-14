"""The analysis pipeline, one span per stage:

    load → preprocess → redact → agent → verify → gate → sanitize → notify

All I/O goes through the adapters passed in; the agent is injected too (the stub
for dry runs). Stage order is the security boundary: the agent only ever sees
redacted text, and the notifier only ever sees a verified, gated and sanitized
diagnosis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from opentelemetry import trace

import telemetry
from adapters.base import (
    AdapterError,
    LogSource,
    Notifier,
    RepoReader,
    Report,
    RunDetails,
    RunRef,
)
from agent.base import AgentRequest, AgentUsage, DiagnosisAgent
from agent.schema import Diagnosis
from guardrails.budget import Budget, BudgetLimits
from guardrails.gating import AlertLevel, GateDecision, apply_auto_fix_gating, decide
from guardrails.preprocess import Excerpt, FatalError, LogLine, preprocess
from guardrails.redact import redact_with_report
from guardrails.sanitize import sanitize_diagnosis
from guardrails.verify_evidence import FileReader, verify_evidence


@dataclass(frozen=True)
class PipelineResult:
    diagnosis: Diagnosis  # verified, gated, sanitized: exactly what a notifier receives
    decision: GateDecision
    run: RunDetails
    dropped_evidence: int  # cited by the agent but failed verification
    redactions: dict[str, int]
    usage: AgentUsage
    notified: bool


# ------------------------------------------------------------------ helpers


def redact_excerpt(excerpt: Excerpt) -> tuple[Excerpt, dict[str, int]]:
    """Redact the kept lines as one text (so multi-line keys are caught), keeping numbering."""
    result = redact_with_report("\n".join(line.text for line in excerpt.lines))
    texts = result.text.split("\n")
    if len(texts) != len(excerpt.lines):  # redaction is line-preserving; never trust blindly
        raise RuntimeError("redaction changed the number of lines")
    by_number = {line.number: t for line, t in zip(excerpt.lines, texts, strict=True)}

    def fix(err: FatalError | None) -> FatalError | None:
        if err is None:
            return None
        text = by_number.get(err.line.number)
        if text is None:  # always kept by preprocess; redact on its own just in case
            text = redact_with_report(err.line.text).text
        return replace(err, line=LogLine(number=err.line.number, text=text))

    redacted = replace(
        excerpt,
        lines=tuple(LogLine(number=n, text=t) for n, t in by_number.items()),
        first_error=fix(excerpt.first_error),
        termination=fix(excerpt.termination),
    )
    return redacted, result.counts


def run_attempts_text(run: RunDetails) -> str:
    """The ``run_attempts`` context block, as an agent sees it and the verifier checks it."""
    lines = [f"attempt {a.attempt}: {a.conclusion or 'unknown'}" for a in run.attempts]
    lines += [
        f"related run {r.run_id} ({r.build_tool}, {r.event}): {r.conclusion}"
        for r in run.related_runs
    ]
    return "\n".join(lines)


def file_reader(repo: RepoReader | None, sha: str) -> FileReader:
    """Files at the failing commit for the verifier; unreadable means unverifiable."""

    def read(path: str) -> str | None:
        if repo is None:
            return None
        try:
            return repo.read_file(sha, path)
        except AdapterError:
            return None

    return read


def _error_attrs(prefix: str, err: FatalError | None) -> Mapping[str, str | int]:
    if err is None:
        return {f"{prefix}.found": "false"}
    return {f"{prefix}.kind": err.kind.value, f"{prefix}.line": err.line.number}


# ---------------------------------------------------------------- pipeline


def analyze(
    ref: RunRef,
    *,
    logs: LogSource,
    repo: RepoReader | None,
    notifier: Notifier,
    agent: DiagnosisAgent,
    tracer: trace.Tracer | None = None,
    limits: BudgetLimits | None = None,
) -> PipelineResult:
    tracer = tracer or telemetry.get_tracer()
    budget = Budget(limits)

    with tracer.start_as_current_span("analyze") as root:
        root.set_attributes({"repo": ref.repo, "run_id": ref.run_id, "agent": agent.name})

        with tracer.start_as_current_span("load") as span:
            run = logs.get_run(ref)
            raw = logs.get_log(ref)
            span.set_attributes(
                {
                    "build_tool": run.build_tool,
                    "event": run.event,
                    "head_sha": run.head_sha,
                    "conclusion": run.conclusion,
                    "attempts": len(run.attempts),
                    "log_bytes": len(raw.encode()),
                }
            )
            root.set_attribute("head_sha", run.head_sha)

        with tracer.start_as_current_span("preprocess") as span:
            excerpt = preprocess(raw)
            span.set_attributes(
                {
                    "raw_lines": excerpt.total_lines,
                    "kept_lines": len(excerpt.lines),
                    **_error_attrs("first_error", excerpt.first_error),
                    **_error_attrs("termination", excerpt.termination),
                }
            )

        with tracer.start_as_current_span("redact") as span:
            excerpt, redactions = redact_excerpt(excerpt)
            span.set_attribute("redacted", sum(redactions.values()))
            span.set_attributes({f"redacted.{k}": v for k, v in sorted(redactions.items())})

        with tracer.start_as_current_span("agent") as span:
            result = agent.diagnose(AgentRequest(run=run, excerpt=excerpt), budget)
            budget.check_time()
            usage = result.usage
            span.set_attributes(
                {
                    "agent": agent.name,
                    "model_id": usage.model_id or "none",
                    "llm_calls": usage.llm_calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": usage.cost_usd,
                    "status": result.diagnosis.status.value,
                    "category": result.diagnosis.category.value,
                    "confidence": result.diagnosis.confidence,
                    "evidence.cited": len(result.diagnosis.evidence),
                }
            )

        with tracer.start_as_current_span("verify") as span:
            verified = verify_evidence(
                result.diagnosis,
                log=raw,
                read_file=file_reader(repo, run.head_sha),
                context={"run_attempts": run_attempts_text(run)},
            )
            diagnosis = verified.diagnosis
            span.set_attributes(
                {
                    "evidence.cited": verified.cited,
                    "evidence.verified": len(diagnosis.evidence),
                    "evidence.dropped": len(verified.dropped),
                    "repo_reader": repo is not None,
                    "status": diagnosis.status.value,
                    "confidence": diagnosis.confidence,
                }
            )

        with tracer.start_as_current_span("gate") as span:
            diagnosis = apply_auto_fix_gating(diagnosis)
            decision = decide(diagnosis)
            span.set_attributes(
                {
                    "alert_level": decision.level.value,
                    "channels": [c.value for c in decision.channels],
                    "auto_fixable": diagnosis.auto_fixable,
                }
            )

        with tracer.start_as_current_span("sanitize") as span:
            before = diagnosis
            diagnosis = sanitize_diagnosis(diagnosis, repo=ref.repo)
            span.set_attribute("changed", diagnosis != before)

        with tracer.start_as_current_span("notify") as span:
            notified = decision.level is not AlertLevel.none
            if notified:
                notifier.notify(
                    Report(
                        repo=ref.repo,
                        head_sha=run.head_sha,
                        build_tools=(run.build_tool,),
                        diagnosis=diagnosis,
                    )
                )
            span.set_attributes({"notified": notified, "alert_level": decision.level.value})

        root.set_attributes(
            {
                "status": diagnosis.status.value,
                "category": diagnosis.category.value,
                "confidence": diagnosis.confidence,
                "alert_level": decision.level.value,
                "elapsed_s": budget.elapsed(),
            }
        )

    return PipelineResult(
        diagnosis=diagnosis,
        decision=decision,
        run=run,
        dropped_evidence=len(verified.dropped),
        redactions=redactions,
        usage=usage,
        notified=notified,
    )
