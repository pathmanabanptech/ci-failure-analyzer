"""Notification gating, category evidence requirements and auto-fix gating.

Runs after the evidence and fix verifiers, so ``confidence``, ``verified`` and
``fix_diff`` are already code-checked. Order in the pipeline:

1. ``apply_evidence_requirements``: each category needs specific kinds of verified
   evidence (a compile error needs the compiler's log line, not just a source line).
   If they are missing, a diagnosis is downgraded to ``possible_cause`` with
   confidence capped at 0.6.
2. ``apply_auto_fix_gating``: sets ``auto_fixable``.
3. ``decide``: alert level and channels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from agent.schema import Category, Diagnosis, Evidence, Status

ALERT_THRESHOLD = 0.7
POSSIBLE_CAUSE_THRESHOLD = 0.4
REQUIREMENT_CONFIDENCE_CAP = 0.6
AUTO_FIX_CATEGORIES = frozenset({Category.compile, Category.format, Category.test_assertion})

BUILD_FILES = frozenset(
    {
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradle.properties",
    }
)


# ------------------------------------------------- category evidence requirements


@dataclass(frozen=True)
class RequiredEvidence:
    name: str
    matches: Callable[[Evidence], bool]


LOG_LINE = RequiredEvidence("a verified log line", lambda e: e.kind == "log")
BUILD_FILE = RequiredEvidence(
    "a verified build file (pom.xml, build.gradle, ...)",
    lambda e: e.kind == "file" and PurePosixPath(e.path or "").name in BUILD_FILES,
)
RUN_ATTEMPTS = RequiredEvidence("verified run attempts", lambda e: e.kind == "run_attempts")

# Every kind listed must be present among the verified evidence. ``unknown`` has no
# requirement that can be met: an unknown cause is never a diagnosis.
REQUIREMENTS: dict[Category, tuple[RequiredEvidence, ...] | None] = {
    Category.compile: (LOG_LINE,),  # the compiler error itself
    Category.dependency: (LOG_LINE, BUILD_FILE),  # resolution error + the declaration
    Category.format: (LOG_LINE,),
    Category.test_assertion: (LOG_LINE,),
    Category.flaky: (LOG_LINE, RUN_ATTEMPTS),  # fails, then passes on re-run
    Category.memory: (LOG_LINE,),
    Category.config: (LOG_LINE,),
    Category.dependency_service: (LOG_LINE,),
    Category.timeout: (LOG_LINE,),
    Category.runner: (LOG_LINE,),
    Category.cancelled: (LOG_LINE,),
    Category.unknown: None,
}


@dataclass(frozen=True)
class RequirementCheck:
    met: bool
    missing: tuple[str, ...]  # human-readable names of the missing evidence kinds


def check_evidence_requirements(d: Diagnosis) -> RequirementCheck:
    kinds = REQUIREMENTS[d.category]
    if kinds is None:
        return RequirementCheck(False, ("a known category (unknown cannot be diagnosed)",))
    verified = [e for e in d.evidence if e.verified]
    missing = tuple(k.name for k in kinds if not any(k.matches(e) for e in verified))
    return RequirementCheck(not missing, missing)


def apply_evidence_requirements(d: Diagnosis) -> tuple[Diagnosis, RequirementCheck]:
    """Downgrade to ``possible_cause`` (confidence <= 0.6) if requirements are unmet.

    ``undiagnosable`` stays as it is; nothing is ever upgraded.
    """
    check = check_evidence_requirements(d)
    if check.met or d.status is Status.undiagnosable:
        return d, check
    update = {
        "status": Status.possible_cause,
        "confidence": min(d.confidence, REQUIREMENT_CONFIDENCE_CAP),
    }
    return d.model_copy(update=update), check


# ------------------------------------------------------------------- alerts


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


# ----------------------------------------------------------------- auto-fix


def is_auto_fixable(d: Diagnosis, *, fix_verified: bool) -> bool:
    """Allowlisted category, >= 1 verified evidence, confidence >= 0.7, diagnosed, and a
    ``fix_diff`` that the fix verifier confirmed applies (nothing to hand off otherwise)."""
    return (
        d.category in AUTO_FIX_CATEGORIES
        and any(e.verified for e in d.evidence)
        and d.confidence >= ALERT_THRESHOLD
        and d.status is Status.diagnosed
        and d.fix_diff is not None
        and fix_verified
    )


def apply_auto_fix_gating(d: Diagnosis, *, fix_verified: bool) -> Diagnosis:
    """Set ``auto_fixable`` by code, overriding whatever the model said.

    ``fix_verified`` must come from the fix verifier (``git apply --check`` passed).
    """
    return d.model_copy(update={"auto_fixable": is_auto_fixable(d, fix_verified=fix_verified)})
