"""Budgets: tool calls per agent, input tokens, wall clock; cost accounting.

No I/O; the clock is injectable. Exceeding any limit raises ``BudgetExceeded``,
which the pipeline turns into a fail-safe exit (post nothing, exit 0), never a retry.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
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

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BudgetLimits:
        """Defaults, overridable with BUDGET_MAX_TOOL_CALLS, BUDGET_MAX_INPUT_TOKENS and
        BUDGET_TIMEOUT_S (e.g. a slower local model). Invalid values raise ValueError."""
        env = os.environ if env is None else env
        d = cls()

        def num(name: str, default: float) -> float:
            raw = env.get(name)
            if not raw:
                return default
            value = float(raw)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {raw!r}")
            return value

        return cls(
            max_tool_calls_per_agent=int(num("BUDGET_MAX_TOOL_CALLS", d.max_tool_calls_per_agent)),
            max_input_tokens=int(num("BUDGET_MAX_INPUT_TOKENS", d.max_input_tokens)),
            timeout_s=num("BUDGET_TIMEOUT_S", d.timeout_s),
        )


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
