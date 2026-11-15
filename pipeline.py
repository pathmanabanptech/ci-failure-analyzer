"""The analysis pipeline, one span per stage:

    load → preprocess → redact → agent → verify → gate → sanitize → notify

``verify`` checks cited evidence and, separately, that ``fix_diff`` applies to what
the build compiled (the merge for pull_request builds, head otherwise); a fix that
does not apply is dropped.

All I/O goes through the adapters passed in; the agent is injected too (the
generalist, or the stub for dry runs). Stage order is the security boundary: the
agent only ever sees redacted text, and the notifier only ever sees a verified,
gated and sanitized diagnosis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from agent.base import AgentRequest, AgentResult, AgentUsage, DiagnosisAgent, run_attempts_text
from agent.schema import Diagnosis, Evidence, Ref
from guardrails.budget import Budget, BudgetLimits
from guardrails.gating import (
    AlertLevel,
    GateDecision,
    RequirementCheck,
    apply_auto_fix_gating,
    apply_evidence_requirements,
    decide,
)
from guardrails.preprocess import Excerpt, FatalError, LogLine, clean_log, preprocess
from guardrails.redact import redact, redact_log_lines

__all__ = ["PipelineResult", "analyze", "file_reader", "redact_excerpt", "run_attempts_text"]
from guardrails.sanitize import sanitize_diagnosis
from guardrails.verify_evidence import (
    EvidenceCheck,
    FileReader,
    default_ref_for,
    verify_evidence,
)
from guardrails.verify_fix import FixCheck, PatchChecker, verify_fix


@dataclass(frozen=True)
class PipelineResult:
    diagnosis: Diagnosis  # verified, gated, sanitized: exactly what a notifier receives
    decision: GateDecision
    run: RunDetails
    redactions: dict[str, int]
    usage: AgentUsage
    notified: bool
    agent_output: AgentResult  # the agent's result before any guardrail (debugging)
    evidence_checks: tuple[EvidenceCheck, ...]  # verifier verdict + reason per cited item
    fix_check: FixCheck | None  # None: the agent proposed no fix
    requirements: RequirementCheck  # category evidence requirements

    @property
    def dropped_evidence(self) -> tuple[Evidence, ...]:
        """Cited by the agent but failed verification."""
        return tuple(c.evidence for c in self.evidence_checks if not c.ok)


# ------------------------------------------------------------------ helpers


def redact_excerpt(
    excerpt: Excerpt, log_lines: Sequence[LogLine]
) -> tuple[Excerpt, tuple[LogLine, ...], dict[str, int]]:
    """Redact the whole cleaned log once; the excerpt takes its lines from the result."""
    redacted_log, counts = redact_log_lines(log_lines)
    by_number = {line.number: line for line in redacted_log}

    def pick(line: LogLine) -> LogLine:
        return by_number.get(line.number) or redact_log_lines([line])[0][0]

    def fix(err: FatalError | None) -> FatalError | None:
        return None if err is None else replace(err, line=pick(err.line))

    redacted = replace(
        excerpt,
        lines=tuple(pick(line) for line in excerpt.lines),
        first_error=fix(excerpt.first_error),
        termination=fix(excerpt.termination),
    )
    return redacted, redacted_log, counts


def file_reader(repo: RepoReader | None, run: RunDetails) -> FileReader:
    """Files for the verifier at a given ref, redacted like the agent saw them;
    unreadable means unverifiable. ``merge`` on a push build is head (what CI built)."""

    def read(path: str, ref: Ref) -> str | None:
        if repo is None:
            return None
        try:
            if ref == "base":
                return redact(repo.read_file(run.base_sha, path)) if run.base_sha else None
            if ref == "merge" and run.base_sha is not None:
                return redact(repo.read_file_at_merge(run.base_sha, run.head_sha, path))
            return redact(repo.read_file(run.head_sha, path))
        except AdapterError:
            return None

    return read


def patch_checker(repo: RepoReader | None, run: RunDetails) -> PatchChecker | None:
    """Check fixes against what the build compiled: the merge for PR builds, else head."""
    if repo is None:
        return None
    base = run.base_sha if default_ref_for(run.event) == "merge" else None

    def check(patch: str) -> FixCheck:
        try:
            result = repo.check_patch(patch, run.head_sha, base)
        except AdapterError as e:
            return FixCheck(False, str(e))
        return FixCheck(result.ok, result.message)

    return check


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
    log_name: str | None = None,
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
            excerpt, log_lines, redactions = redact_excerpt(excerpt, clean_log(raw)[0])
            span.set_attribute("redacted", sum(redactions.values()))
            span.set_attributes({f"redacted.{k}": v for k, v in sorted(redactions.items())})

        with tracer.start_as_current_span("agent") as span:
            request = AgentRequest(run=run, excerpt=excerpt, log_lines=log_lines, log_name=log_name)
            result = agent.diagnose(request, budget)
            budget.check_time()
            usage = result.usage
            span.set_attributes(
                {
                    "agent": agent.name,
                    "model_id": usage.model_id or "none",
                    "llm_calls": usage.llm_calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": "unknown" if usage.cost_usd is None else usage.cost_usd,
                    "tool_calls": result.tool_calls,
                    "structured_output_fallback": result.structured_output_fallback,
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
                read_file=file_reader(repo, run),
                context={**result.context, "run_attempts": run_attempts_text(run)},
                default_ref=default_ref_for(run.event),
                log_name=log_name,
            )
            diagnosis, fix_check = verify_fix(verified.diagnosis, patch_checker(repo, run))
            span.set_attributes(
                {
                    "evidence.cited": verified.cited,
                    "evidence.verified": len(diagnosis.evidence),
                    "evidence.dropped": len(verified.dropped),
                    "fix.proposed": fix_check is not None,
                    "fix.applies": bool(fix_check and fix_check.ok),
                    "repo_reader": repo is not None,
                    "status": diagnosis.status.value,
                    "confidence": diagnosis.confidence,
                }
            )

        with tracer.start_as_current_span("gate") as span:
            diagnosis, requirements = apply_evidence_requirements(diagnosis)
            fix_verified = fix_check is not None and fix_check.ok
            diagnosis = apply_auto_fix_gating(diagnosis, fix_verified=fix_verified)
            decision = decide(diagnosis)
            span.set_attributes(
                {
                    "evidence_requirements.met": requirements.met,
                    "evidence_requirements.missing": len(requirements.missing),
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
        agent_output=result,
        evidence_checks=verified.checks,
        fix_check=fix_check,
        requirements=requirements,
        redactions=redactions,
        usage=usage,
        notified=notified,
    )
