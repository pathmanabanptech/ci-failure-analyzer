"""G3 secret redaction.

Deterministic, gitleaks-style pass applied before content reaches the model, before
it is stored in memory and before a notification is posted.

Properties (all tested):

* **Stable placeholders**: ``[REDACTED:<kind>]``, so the model still sees *that*
  something was there and what kind it was.
* **Idempotent**: ``redact(redact(x)) == redact(x)``; placeholders never re-match.
* **Line-preserving**: the number and order of lines never change (a multi-line
  private key becomes one placeholder per line), so original log line numbers stay
  valid for evidence verification (G5).
* Works on raw logs (runner timestamp prefixes) and on cleaned text alike.

Already-masked values (GitHub's ``***``) and variable references (``$VAR``,
``${VAR}``) are left alone: they are not secrets.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

PLACEHOLDER = "[REDACTED:{kind}]"
_PLACEHOLDER_RE = re.compile(r"\[REDACTED:[a-z_]+\]")

# A runner timestamp prefix (raw logs) at the start of a line, kept inside key blocks.
_LINE_PREFIX = re.compile(r"^(?:﻿)?(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?)?")


def _ph(kind: str) -> str:
    return PLACEHOLDER.format(kind=kind)


# ------------------------------------------------------------------ private keys

_KEY_BEGIN = r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
_KEY_END = r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"
_PRIVATE_KEY_BLOCK = re.compile(f"{_KEY_BEGIN}.*?{_KEY_END}", re.S)
# BEGIN without END (e.g. a truncated log): the BEGIN line plus following base64 lines.
_TS = r"(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?)?"
_B64_LINE = rf"\n{_TS}\s*[A-Za-z0-9+/=]{{16,}}[ \t]*(?=\n|$)"
_PRIVATE_KEY_OPEN = re.compile(rf"{_KEY_BEGIN}[^\n]*(?:{_B64_LINE})*")


def _redact_block(match: re.Match[str]) -> str:
    """One placeholder per line of the match, keeping each line's timestamp prefix."""
    out = []
    for i, segment in enumerate(match.group(0).split("\n")):
        prefix = "" if i == 0 else _LINE_PREFIX.match(segment).group(0)  # type: ignore[union-attr]
        out.append(prefix + _ph("private_key"))
    return "\n".join(out)


# ---------------------------------------------------------- single-line patterns


@dataclass(frozen=True)
class _Rule:
    kind: str
    pattern: re.Pattern[str]
    # Group that holds the secret; everything else in the match is kept.
    group: int = 0


_VALUE = r"""("[^"\n]*"|'[^'\n]*'|[^\s;&,"'<>]+)"""
_NOT_A_SECRET = re.compile(
    r"^(?:\*+|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_]+%|\[REDACTED:[a-z_]+\])$"
)

_RULES: tuple[_Rule, ...] = (
    _Rule(
        "aws_key",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AROA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    _Rule(
        "aws_secret",
        re.compile(
            r"(?i)(?:aws_?secret_?access_?key|aws_?secret_?key|secret_?access_?key)"
            r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
        ),
        group=1,
    ),
    _Rule(
        "github_token",
        re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,255}(?![A-Za-z0-9_])"),
    ),
    _Rule(
        "github_token",
        re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{22,255}(?![A-Za-z0-9_])"),
    ),
    _Rule(
        "jwt",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
            r"(?![A-Za-z0-9_-])"
        ),
    ),
    # scheme://user:password@host  -> scheme://[REDACTED:connection_string]@host
    _Rule(
        "connection_string",
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://([^\s/:@\[\]]+:(?!\*+@)[^\s/@\[\]]+)@"),
        group=1,
    ),
    # password=..., DB_PASSWORD=..., spring.datasource.password=..., --password=...
    # Only with '=' (``server-password: GITHUB_TOKEN`` in setup-java inputs is a name).
    _Rule(
        "password",
        re.compile(
            r"(?i)(?<![A-Za-z0-9])[A-Za-z0-9_.-]*(?:password|passwd|passphrase)\s*=\s*" + _VALUE
        ),
        group=1,
    ),
)


def _make_replacer(rule: _Rule, counts: Counter[str]) -> Callable[[re.Match[str]], str]:
    def replace(m: re.Match[str]) -> str:
        secret = m.group(rule.group)
        if _NOT_A_SECRET.match(secret) or _PLACEHOLDER_RE.fullmatch(secret):
            return m.group(0)
        counts[rule.kind] += 1
        if rule.group == 0:
            return _ph(rule.kind)
        start, end = m.start(rule.group) - m.start(), m.end(rule.group) - m.start()
        whole = m.group(0)
        return whole[:start] + _ph(rule.kind) + whole[end:]

    return replace


# ------------------------------------------------------------------------- API


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)  # kind -> number redacted

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact_with_report(text: str) -> RedactionResult:
    counts: Counter[str] = Counter()

    def block(m: re.Match[str]) -> str:
        counts["private_key"] += 1
        return _redact_block(m)

    text = _PRIVATE_KEY_BLOCK.sub(block, text)
    text = _PRIVATE_KEY_OPEN.sub(block, text)
    for rule in _RULES:
        text = rule.pattern.sub(_make_replacer(rule, counts), text)
    return RedactionResult(text=text, counts=dict(counts))


def redact(text: str) -> str:
    return redact_with_report(text).text
