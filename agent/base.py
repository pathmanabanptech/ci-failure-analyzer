"""Agent input contract and the interface every diagnosing agent implements.

The pipeline builds an ``AgentRequest`` after preprocessing and secret redaction,
so nothing an agent receives here still contains a secret. The excerpt and any
content an agent later fetches through tools are untrusted; wrapping them in
delimited blocks for the model is the agent's job.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from adapters.base import RunDetails
from agent.schema import Diagnosis
from guardrails.budget import Budget
from guardrails.preprocess import Excerpt, LogLine


@dataclass(frozen=True)
class AgentUsage:
    """What one diagnosis cost. A deterministic agent reports zeros."""

    model_id: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None: the provider's price is not configured


@dataclass(frozen=True)
class AgentRequest:
    run: RunDetails  # API metadata only: no branch names, PR titles or answer fields
    excerpt: Excerpt  # preprocessed + redacted; line numbers are raw-log line numbers
    # Every cleaned + redacted log line (raw numbering), for reading outside the excerpt.
    log_lines: tuple[LogLine, ...] = ()
    log_name: str | None = None  # which job log this is, e.g. "maven.log"


@dataclass(frozen=True)
class ToolCall:
    """One tool call as the agent made it (for debugging and eval metrics)."""

    name: str
    args: Mapping[str, Any]  # model-chosen arguments: untrusted
    result_chars: int  # size of the result text the model received
    status: str  # "success" | "error" | "cancelled" (e.g. tool budget used up)


@dataclass(frozen=True)
class AgentResult:
    diagnosis: Diagnosis  # as returned by the agent; not yet verified, gated or sanitized
    usage: AgentUsage = field(default_factory=AgentUsage)
    # Context blocks the agent was shown (e.g. "commit": the commit list), so evidence
    # citing them can be verified against exactly what the agent saw.
    context: Mapping[str, str] = field(default_factory=dict)
    tool_calls: int = 0
    structured_output_fallback: bool = False  # answer came from the constrained retry
    # The model's output exactly as returned (JSON), before any guardrail; for debugging.
    raw_output: str | None = None
    tool_log: tuple[ToolCall, ...] = ()  # every tool call, in order


# The only run metadata an agent sees, from meta.yaml / the CI API. An allowlist, so a
# new metadata field never reaches a model by accident. Never: branch names, PR
# titles or descriptions, anything from the answer key.
RUN_CONTEXT_FIELDS = (
    "build_tool",
    "event",
    "head_sha",
    "base_sha",
    "conclusion",
    "attempts",
    "related_runs",
)
RELATED_RUN_FIELDS = ("build_tool", "event", "conclusion", "base_sha")


def run_context(run: RunDetails) -> dict[str, Any]:
    """Allowlisted run metadata for the agent (see RUN_CONTEXT_FIELDS)."""
    return {
        "build_tool": run.build_tool,
        "event": run.event,
        "head_sha": run.head_sha,
        "base_sha": run.base_sha,
        "conclusion": run.conclusion,
        "attempts": len(run.attempts),
        "related_runs": [
            {
                "build_tool": r.build_tool,
                "event": r.event,
                "conclusion": r.conclusion,
                "base_sha": r.base_sha,
            }
            for r in run.related_runs
        ],
    }


def run_attempts_text(run: RunDetails) -> str:
    """The ``run_attempts`` block, as an agent sees it and the verifier checks it."""
    lines = [f"attempt {a.attempt}: {a.conclusion or 'unknown'}" for a in run.attempts]
    lines += [
        f"related run {r.run_id} ({r.build_tool}, {r.event}): {r.conclusion}"
        for r in run.related_runs
    ]
    return "\n".join(lines)


class DiagnosisAgent(Protocol):
    name: str

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        """Charge ``budget`` before each tool or model call; let BudgetExceeded propagate."""
        ...
