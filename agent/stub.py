"""Deterministic stand-in for the LLM agent (dry runs, pipeline tests).

Makes no model call and no tool call. It returns the same hedged diagnosis for
every log, citing the excerpt's first fatal error line as its only evidence, so a
dry run exercises the downstream stages (the cited line goes through the real
evidence verifier, then gating, sanitizing and notifying).
"""

from __future__ import annotations

from agent.base import AgentRequest, AgentResult, AgentUsage
from agent.schema import Category, Diagnosis, Evidence, Status
from guardrails.budget import Budget

STUB_CONFIDENCE = 0.5


class StubAgent:
    name = "stub"

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        budget.check_time()
        first = request.excerpt.first_error
        if first is None:
            diagnosis = Diagnosis(
                status=Status.undiagnosable,
                category=Category.unknown,
                specialist=None,
                root_cause="[dry run] No fatal error line found; no model was called.",
                local_vs_ci="[dry run] Not analyzed.",
                evidence=[],
                fix_diff=None,
                confidence=0.0,
                build_tool=request.run.build_tool,
            )
        else:
            diagnosis = Diagnosis(
                status=Status.possible_cause,
                category=Category.unknown,
                specialist=None,
                root_cause=(
                    f"[dry run] First fatal error ({first.kind.value}) at log line "
                    f"{first.line.number}; no model was called."
                ),
                local_vs_ci="[dry run] Not analyzed.",
                evidence=[
                    Evidence(
                        kind="log",
                        log=request.log_name,
                        line=first.line.number,
                        snippet=first.line.text,
                    )
                ],
                fix_diff=None,
                confidence=STUB_CONFIDENCE,
                build_tool=request.run.build_tool,
            )
        return AgentResult(
            diagnosis=diagnosis,
            usage=AgentUsage(model_id=None),
            raw_output=diagnosis.model_dump_json(indent=2),
        )
