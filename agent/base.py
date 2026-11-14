"""Agent input contract and the interface every diagnosing agent implements.

The pipeline builds an ``AgentRequest`` after preprocessing and secret redaction,
so nothing an agent receives here still contains a secret. The excerpt and any
content an agent later fetches through tools are untrusted; wrapping them in
delimited blocks for the model is the agent's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from adapters.base import RunDetails
from agent.schema import Diagnosis
from guardrails.budget import Budget
from guardrails.preprocess import Excerpt


@dataclass(frozen=True)
class AgentUsage:
    """What one diagnosis cost. A deterministic agent reports zeros."""

    model_id: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class AgentRequest:
    run: RunDetails  # API metadata only: no branch names, PR titles or answer fields
    excerpt: Excerpt  # preprocessed + redacted; line numbers are raw-log line numbers


@dataclass(frozen=True)
class AgentResult:
    diagnosis: Diagnosis  # as returned by the agent; not yet verified, gated or sanitized
    usage: AgentUsage = field(default_factory=AgentUsage)


class DiagnosisAgent(Protocol):
    name: str

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        """Charge ``budget`` before each tool or model call; let BudgetExceeded propagate."""
        ...
