"""Budgets: tool calls per agent, input tokens, wall clock; cost accounting.

No I/O; the clock is injectable. Exceeding any limit raises ``BudgetExceeded``,
which the pipeline turns into a fail-safe exit (post nothing, exit 0), never a retry.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

BudgetKind = Literal["tool_calls", "input_tokens", "wall_clock"]


class BudgetExceeded(Exception):
    def __init__(self, kind: BudgetKind, limit: float, used: float, agent: str | None = None):
        self.kind, self.limit, self.used, self.agent = kind, limit, used, agent
        who = f" by agent {agent!r}" if agent else ""
        super().__init__(f"{kind} budget exceeded{who}: {used:g} > {limit:g}")


@dataclass(frozen=True)
class BudgetLimits:
    max_tool_calls_per_agent: int = 8
    max_input_tokens: int = 60_000
    timeout_s: float = 120.0


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


@dataclass(frozen=True)
class Usage:
    tool_calls: dict[str, int]
    input_tokens: int
    output_tokens: int
    elapsed_s: float


class Budget:
    """Per-diagnosis budget. Charge before doing the work."""

    def __init__(
        self, limits: BudgetLimits | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limits = limits or BudgetLimits()
        self._clock = clock
        self._start = clock()
        self._tool_calls: Counter[str] = Counter()
        self._input_tokens = 0
        self._output_tokens = 0

    def elapsed(self) -> float:
        return self._clock() - self._start

    def check_time(self) -> None:
        if (elapsed := self.elapsed()) > self.limits.timeout_s:
            raise BudgetExceeded("wall_clock", self.limits.timeout_s, elapsed)

    def charge_tool_call(self, agent: str = "generalist") -> None:
        self.check_time()
        used = self._tool_calls[agent] + 1
        if used > self.limits.max_tool_calls_per_agent:
            raise BudgetExceeded("tool_calls", self.limits.max_tool_calls_per_agent, used, agent)
        self._tool_calls[agent] = used

    def charge_input(self, tokens: int) -> None:
        """Reserve input tokens for a model call before making it."""
        self.check_time()
        used = self._input_tokens + tokens
        if used > self.limits.max_input_tokens:
            raise BudgetExceeded("input_tokens", self.limits.max_input_tokens, used)
        self._input_tokens = used

    def record_output(self, tokens: int) -> None:
        self._output_tokens += tokens

    def usage(self) -> Usage:
        return Usage(
            tool_calls=dict(self._tool_calls),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            elapsed_s=self.elapsed(),
        )

    def cost(self, price: Price) -> float:
        return price.cost(self._input_tokens, self._output_tokens)
