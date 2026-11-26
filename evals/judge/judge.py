"""LLM judge for root cause and local_vs_ci quality.

The judge reads ``rubric.md`` (its system prompt), the answer key's ``key_facts`` and
the diagnosis's two prose fields, and returns two scores with a one-line reason each.
The diagnosis is wrapped as untrusted data: it was written by the agent from CI logs,
so it can carry injected instructions aimed at the judge.

Configured separately from the agent, so the judge can be a different model from the
one being evaluated (the eval runner warns when they are the same):

    JUDGE_MODEL_PROVIDER   ollama | bedrock | anthropic
    JUDGE_MODEL_ID         pinned model id, same rules as SPECIALIST_MODEL_ID
    JUDGE_MODEL_PRICE_INPUT_PER_MTOK / JUDGE_MODEL_PRICE_OUTPUT_PER_MTOK   optional

Other model settings (timeouts, Ollama host, API keys) are shared with the agent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent.errors import ModelConfigError, ModelUnavailableError
from agent.schema import Diagnosis
from evals.loader import AnswerKey

RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.md"
JUDGE_ENV = ("JUDGE_MODEL_PROVIDER", "JUDGE_MODEL_ID")


class JudgeOutput(BaseModel):
    """What the judge model must return."""

    root_cause_score: int = Field(ge=0, le=3)
    root_cause_reason: str
    local_vs_ci_score: int = Field(ge=0, le=2)
    local_vs_ci_reason: str


@dataclass(frozen=True)
class JudgeScore:
    root_cause: int  # 0-3
    local_vs_ci: int  # 0-2
    root_cause_reason: str
    local_vs_ci_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0


class Judge(Protocol):
    name: str  # model id the scores came from (recorded in results)

    def score(self, key: AnswerKey, diagnosis: Diagnosis) -> JudgeScore:
        """Raise ModelError subclasses on failure; the eval records them per run."""
        ...


def rubric_text() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8")


def rubric_hash() -> str:
    return hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()


def render_request(key: AnswerKey, diagnosis: Diagnosis) -> str:
    """The judge's user message. JSON-encoding the diagnosis keeps its text from
    closing the untrusted block or posing as part of the instructions."""
    facts = "\n".join(f"- {f}" for f in key.key_facts)
    under_review = json.dumps(
        {"root_cause": diagnosis.root_cause, "local_vs_ci": diagnosis.local_vs_ci},
        ensure_ascii=True,
        indent=2,
    )
    # json.dumps leaves < and > alone; escape them so no tag can end the block early.
    under_review = under_review.replace("<", "\\u003c").replace(">", "\\u003e")
    return (
        f"<key_facts>\n{facts}\n</key_facts>\n\n"
        f"<diagnosis_under_review>\n{under_review}\n</diagnosis_under_review>\n\n"
        "Grade the diagnosis with the rubric."
    )


class LLMJudge:
    """Scores through a Strands model (any provider ``agent.models`` supports)."""

    def __init__(self, handle: Any) -> None:  # agent.models.ModelHandle
        self.handle = handle
        self.name = handle.pinned_id
        self._rubric = rubric_text()

    def score(self, key: AnswerKey, diagnosis: Diagnosis) -> JudgeScore:
        from strands import Agent  # imported lazily: only real judge runs load the SDK

        agent = Agent(model=self.handle.model, system_prompt=self._rubric, callback_handler=None)
        try:
            result = agent(render_request(key, diagnosis), structured_output_model=JudgeOutput)
            out = result.structured_output
        except Exception as e:  # provider SDKs raise many unrelated types
            raise ModelUnavailableError(
                f"judge call to {self.handle.model_id} failed: {type(e).__name__}: {e}"
            ) from None
        if not isinstance(out, JudgeOutput):
            raise ModelUnavailableError("the judge returned no structured scores")
        usage: Mapping[str, Any] = result.metrics.accumulated_usage
        tokens_in, tokens_out = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
        price = self.handle.price
        return JudgeScore(
            root_cause=out.root_cause_score,
            local_vs_ci=out.local_vs_ci_score,
            root_cause_reason=out.root_cause_reason,
            local_vs_ci_reason=out.local_vs_ci_reason,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=price.cost(tokens_in, tokens_out) if price else None,
        )


def judge_settings_env(env: Mapping[str, str]) -> dict[str, str]:
    """The agent's model settings with the judge's provider, model and price swapped in,
    so the judge reuses ``agent.models`` validation (pinned ids, reachability)."""
    missing = [n for n in JUDGE_ENV if not env.get(n)]
    if missing:
        raise ModelConfigError(f"{' and '.join(missing)} not set (needed for --judge)")
    out = {k: v for k, v in env.items() if not k.startswith("MODEL_PRICE_")}
    out["MODEL_PROVIDER"] = env["JUDGE_MODEL_PROVIDER"]
    out["SPECIALIST_MODEL_ID"] = env["JUDGE_MODEL_ID"]
    for side in ("INPUT", "OUTPUT"):
        if value := env.get(f"JUDGE_MODEL_PRICE_{side}_PER_MTOK"):
            out[f"MODEL_PRICE_{side}_PER_MTOK"] = value
    return out


def judge_from_env(env: Mapping[str, str]) -> LLMJudge:
    from agent.models import ModelSettings, build_model  # loads a model SDK

    settings = ModelSettings.from_env(judge_settings_env(env))
    return LLMJudge(build_model(settings, "specialist"))
