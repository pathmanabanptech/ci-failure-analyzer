"""Single generalist agent: one Strands agent with the four read-only tools.

* The prompt comes from ``prompts/generalist.md``; the run is described with metadata
  and the redacted excerpt inside an ``<untrusted_log>`` block.
* Structured output: the model fills ``ModelDiagnosis`` (no ``verified`` /
  ``auto_fixable``: those are set by code), converted to ``Diagnosis`` here.
* If the model ends without calling the structured-output tool (small local models
  do), one more call asks for the answer through the provider's structured-output
  API (constrained decoding on Ollama). ``AgentResult.structured_output_fallback``
  records it.
* Budgets are enforced through Strands hooks: wall clock before every model and tool
  call, input tokens after every model call (both abort with ``BudgetExceeded``).
  When the tool-call budget is used up, further tool calls are cancelled with a
  message telling the model to answer with what it has.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from strands import Agent
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.types.exceptions import StructuredOutputException

from adapters.base import RepoReader
from agent.base import AgentRequest, AgentResult, AgentUsage, ToolCall, run_context
from agent.models import ModelHandle, ModelUnavailableError
from agent.schema import Category, Diagnosis, Evidence, EvidenceKind, Ref, Status
from agent.tools import ToolContext, build_tools, untrusted
from guardrails.budget import Budget, BudgetExceeded

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generalist.md"
AGENT_NAME = "generalist"
FINAL_ANSWER_PROMPT = (
    "Give your final diagnosis now as a JSON object matching the schema. Cite only "
    "evidence you were actually shown, with each snippet copied exactly. Include the "
    'first fatal log line as kind "log" with its line number. For kind "file" give '
    'path and ref; for kind "commit" give sha.'
)
TOOL_BUDGET_MESSAGE = (
    "Tool budget used up. Do not call more tools; give your final answer now with the "
    "evidence you already have (lower the confidence if the cause is not confirmed)."
)


class ModelEvidence(BaseModel):
    kind: EvidenceKind = Field(
        description='"log" (a log line), "file" (a repository file), "commit" (a commit '
        'shown by recent_commits/symbol_history) or "run_attempts"'
    )
    path: str | None = Field(
        default=None, description='kind "file" only: repo-relative path, e.g. pom.xml'
    )
    ref: Ref | None = Field(
        default=None,
        description='kind "file" only: the version you read, "head", "base" or "merge"',
    )
    sha: str | None = Field(
        default=None, description='kind "commit" only: the commit SHA as shown (12+ hex chars)'
    )
    line: int | None = Field(
        default=None,
        description='kind "log"/"file": the line number shown in the left column (a hint)',
    )
    snippet: str = Field(
        description="the exact text you cite, copied verbatim, without line number or marker"
    )


class ModelDiagnosis(BaseModel):
    """Diagnosis of the CI failure."""

    status: Status
    category: Category
    root_cause: str = Field(description="one or two sentences: the concrete cause")
    local_vs_ci: str = Field(description="why it probably passed on the developer's machine")
    evidence: list[ModelEvidence]
    fix_diff: str | None = Field(default=None, description="minimal unified diff, or null")
    confidence: float = Field(ge=0.0, le=1.0)


# Line-number prefixes as rendered in excerpts/tools ("  277> ", "12| "); models
# sometimes copy them into snippets.
_RENDER_PREFIX = re.compile(r"^\s*\d+[|>!] ?", re.M)


def clean_snippet(snippet: str) -> str:
    return _RENDER_PREFIX.sub("", snippet).strip("\n")


def to_evidence(e: ModelEvidence, request: AgentRequest) -> Evidence:
    """Keep only the locators that belong to the kind; the verifier checks the rest."""
    snippet = clean_snippet(e.snippet)
    if e.kind == "log":
        return Evidence(kind="log", log=request.log_name, line=e.line, snippet=snippet)
    if e.kind == "file":
        path = (e.path or "").strip() or None
        return Evidence(kind="file", path=path, ref=e.ref, line=e.line, snippet=snippet)
    if e.kind == "commit":
        sha = (e.sha or "").strip() or None
        return Evidence(kind="commit", sha=sha, snippet=snippet)
    return Evidence(kind="run_attempts", snippet=snippet)


def to_diagnosis(out: ModelDiagnosis, request: AgentRequest) -> Diagnosis:
    return Diagnosis(
        status=out.status,
        category=out.category,
        specialist=None,  # the generalist is not a specialist
        root_cause=out.root_cause,
        local_vs_ci=out.local_vs_ci,
        evidence=[to_evidence(e, request) for e in out.evidence],
        fix_diff=out.fix_diff or None,
        confidence=out.confidence,
        build_tool=request.run.build_tool,
    )


def render_run_context(request: AgentRequest) -> list[str]:
    ctx = run_context(request.run)
    lines = [
        f"build tool: {ctx['build_tool']}",
        f"event: {ctx['event']}",
        f"head commit (ref=head): {ctx['head_sha']}",
    ]
    if ctx["base_sha"]:
        lines.append(f"PR base commit (ref=base): {ctx['base_sha']}")
    lines.append(f"conclusion: {ctx['conclusion']}, attempts: {ctx['attempts']}")
    related = ctx["related_runs"]
    if related:
        lines.append("other runs of the same head commit:")
        for r in related:
            against = f" against base {r['base_sha']}" if r["base_sha"] else ""
            lines.append(f"  - {r['build_tool']} {r['event']} build{against}: {r['conclusion']}")
    else:
        lines.append("other runs of the same head commit: none")
    return lines


def render_request(request: AgentRequest) -> str:
    ex = request.excerpt
    meta = render_run_context(request)
    if request.log_name:
        meta.insert(0, f"log: {request.log_name}")
    if ex.first_error:
        meta.append(
            f"first fatal error: {ex.first_error.kind.value} at log line "
            f"{ex.first_error.line.number}"
        )
    else:
        meta.append("first fatal error: none detected")
    if ex.termination:
        meta.append(
            f"job ended: {ex.termination.kind.value} at log line {ex.termination.line.number}"
        )
    return (
        "Run context (from the CI system):\n"
        + "\n".join(m if m.startswith("  ") else f"- {m}" for m in meta)
        + f"\n\nLog excerpt ({len(ex.lines)} of {ex.total_lines} lines; "
        "`>` first fatal error, `!` job end):\n"
        + untrusted("log", ex.render())
        + "\n\nDiagnose this failure."
    )


class _BudgetHooks(HookProvider):
    def __init__(self, budget: Budget, agent_name: str) -> None:
        self.budget = budget
        self.agent_name = agent_name
        self.tool_calls = 0
        self.tool_budget_hit = False
        self.estimated_input = 0  # input tokens of calls the SDK does not report
        self._input_seen = 0
        self._last_call_input = 0

    def charge_estimate(self) -> None:
        """Charge a call whose usage is unknown as large as the last model call's input."""
        self.budget.check_time()
        self.budget.charge_input(self._last_call_input)
        self.estimated_input += self._last_call_input

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        self.budget.check_time()

    def _after_model(self, event: AfterModelCallEvent) -> None:
        usage = event.agent.event_loop_metrics.accumulated_usage
        total_in = int(usage.get("inputTokens", 0))
        delta, self._input_seen = total_in - self._input_seen, total_in
        self._last_call_input = max(delta, self._last_call_input)
        if delta > 0:
            self.budget.charge_input(delta)
        self.budget.check_time()

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        try:
            self.budget.charge_tool_call(self.agent_name)
        except BudgetExceeded as e:
            if e.kind != "tool_calls":
                raise
            self.tool_budget_hit = True
            event.cancel_tool = TOOL_BUDGET_MESSAGE
            return
        self.tool_calls += 1


class _ToolRecorder(HookProvider):
    """Records every tool call: name, arguments, result size, outcome."""

    def __init__(self, skip: frozenset[str]) -> None:
        self.calls: list[ToolCall] = []
        self._skip = skip  # the SDK's structured-output tool: its args are the answer

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _after_tool(self, event: AfterToolCallEvent) -> None:
        name = str(event.tool_use.get("name", "?"))
        if name in self._skip:
            return
        result: Mapping[str, Any] = event.result or {}
        size = 0
        for block in result.get("content", []) or []:
            text = block.get("text") if isinstance(block, Mapping) else None
            size += len(text) if isinstance(text, str) else len(json.dumps(block, default=str))
        if event.cancel_message:
            status = "cancelled"
        else:
            status = str(result.get("status", "error" if event.exception else "success"))
        args = event.tool_use.get("input") or {}
        self.calls.append(
            ToolCall(
                name=name,
                args=dict(args) if isinstance(args, Mapping) else {"input": args},
                result_chars=size,
                status=status,
            )
        )


@dataclass(frozen=True)
class PromptInfo:
    text: str
    sha256: str


def load_prompt(path: Path = PROMPT_PATH) -> PromptInfo:
    text = path.read_text(encoding="utf-8")
    return PromptInfo(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


class GeneralistAgent:
    name = AGENT_NAME

    def __init__(
        self, handle: ModelHandle, repo: RepoReader | None, prompt: PromptInfo | None = None
    ) -> None:
        self.handle = handle
        self.repo = repo
        self.prompt = prompt or load_prompt()

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        ctx = ToolContext(run=request.run, log_lines=request.log_lines, repo=self.repo)
        hooks = _BudgetHooks(budget, self.name)
        recorder = _ToolRecorder(skip=frozenset({ModelDiagnosis.__name__}))
        agent = Agent(
            model=self.handle.model,
            system_prompt=self.prompt.text,
            tools=build_tools(ctx),
            hooks=[hooks, recorder],
            callback_handler=None,
            name=self.name,
        )
        fallback = False
        try:
            try:
                result = agent(render_request(request), structured_output_model=ModelDiagnosis)
                out = result.structured_output
            except StructuredOutputException:
                # Small local models sometimes end with prose instead of calling the
                # structured-output tool. One constrained call over the same conversation.
                out, fallback = self._constrained_answer(agent, hooks), True
        except BudgetExceeded:
            raise
        except Exception as e:  # provider/SDK errors of many types; the message is enough
            raise ModelUnavailableError(
                f"{self.handle.provider} agent call to {self.handle.model_id} failed: "
                f"{type(e).__name__}: {e}"
            ) from None

        if not isinstance(out, ModelDiagnosis):
            raise ModelUnavailableError("the model returned no structured diagnosis")
        metrics = agent.event_loop_metrics
        tokens_in = int(metrics.accumulated_usage.get("inputTokens", 0)) + hooks.estimated_input
        tokens_out = int(metrics.accumulated_usage.get("outputTokens", 0))
        budget.record_output(tokens_out)
        price = self.handle.price
        return AgentResult(
            diagnosis=to_diagnosis(out, request),
            usage=AgentUsage(
                model_id=self.handle.pinned_id,
                llm_calls=int(metrics.cycle_count) + int(fallback),
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=price.cost(tokens_in, tokens_out) if price else None,
            ),
            context=ctx.context(),
            tool_calls=hooks.tool_calls,
            structured_output_fallback=fallback,
            raw_output=out.model_dump_json(indent=2),
            tool_log=tuple(recorder.calls),
        )

    def _constrained_answer(self, agent: Agent, hooks: _BudgetHooks) -> ModelDiagnosis:
        """Ask once more for the final answer through the provider's structured-output API
        (Ollama: JSON-schema constrained decoding). No tools; the conversation is reused."""
        hooks.charge_estimate()  # usage of this call is not reported by the SDK
        messages: list[Any] = [
            *agent.messages,
            {"role": "user", "content": [{"text": FINAL_ANSWER_PROMPT}]},
        ]

        async def run() -> ModelDiagnosis:
            out: ModelDiagnosis | None = None
            stream = self.handle.model.structured_output(
                ModelDiagnosis, messages, system_prompt=self.prompt.text
            )
            async for event in stream:
                if isinstance(event.get("output"), ModelDiagnosis):
                    out = event["output"]
            if out is None:
                raise ModelUnavailableError("the model returned no structured diagnosis")
            return out

        return asyncio.run(run())
