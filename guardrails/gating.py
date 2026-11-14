"""Notification gating and deterministic auto-fix gating.

Runs after the evidence verifier, so ``confidence`` and ``verified`` are
already code-checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agent.schema import Category, Diagnosis, Status

ALERT_THRESHOLD = 0.7
POSSIBLE_CAUSE_THRESHOLD = 0.4
AUTO_FIX_CATEGORIES = frozenset({Category.compile, Category.format, Category.test_assertion})


class AlertLevel(StrEnum):
    alert = "alert"  # PR comment + Teams
    possible_cause = "possible_cause"  # PR comment only, labelled "Possible cause"
    none = "none"  # logged only


class Channel(StrEnum):
    pr_comment = "pr_comment"
    teams = "teams"


_CHANNELS = {
    AlertLevel.alert: (Channel.pr_comment, Channel.teams),
    AlertLevel.possible_cause: (Channel.pr_comment,),
    AlertLevel.none: (),
}


@dataclass(frozen=True)
class GateDecision:
    level: AlertLevel
    channels: tuple[Channel, ...]


def alert_level(d: Diagnosis) -> AlertLevel:
    """>= 0.7 alert, 0.4-0.7 possible cause, < 0.4 nothing.

    Only a ``diagnosed`` status can raise a full alert; ``undiagnosable`` never alerts.
    """
    if d.status is Status.undiagnosable or d.confidence < POSSIBLE_CAUSE_THRESHOLD:
        return AlertLevel.none
    if d.confidence >= ALERT_THRESHOLD and d.status is Status.diagnosed:
        return AlertLevel.alert
    return AlertLevel.possible_cause


def decide(d: Diagnosis) -> GateDecision:
    level = alert_level(d)
    return GateDecision(level=level, channels=_CHANNELS[level])


def is_auto_fixable(d: Diagnosis) -> bool:
    """Allowlisted category, >= 1 verified evidence, confidence >= 0.7, diagnosed."""
    return (
        d.category in AUTO_FIX_CATEGORIES
        and any(e.verified for e in d.evidence)
        and d.confidence >= ALERT_THRESHOLD
        and d.status is Status.diagnosed
    )


def apply_auto_fix_gating(d: Diagnosis) -> Diagnosis:
    """Set ``auto_fixable`` by code, overriding whatever the model said."""
    return d.model_copy(update={"auto_fixable": is_auto_fixable(d)})
