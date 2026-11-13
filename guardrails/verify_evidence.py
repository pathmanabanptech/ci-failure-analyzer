"""G5 evidence verifier: code, not the model, decides whether evidence is real.

Every cited snippet is checked at its cited location:

* ``source="log"``: the raw log line ``line`` (1-based, raw numbering as shown in
  the G8 excerpt), cleaned and redacted exactly like the text the model saw. A
  multi-line snippet must match consecutive lines starting at ``line``.
* ``source=<file path>``: the file at the failing commit (via the injected
  ``read_file``). With ``line``: match at that line; without: anywhere in the file.
* ``source="commit"`` / ``"run_attempts"``: the text of that context block.
* Anything else (incl. ``"memory"``: recalled memories are hints, never evidence,
  G12) never verifies.

Matching tolerates whitespace differences and ``...`` elisions; snippets shorter
than ``MIN_SNIPPET_CHARS`` non-space characters never verify (``"error"`` alone
proves nothing).

Unverified evidence is dropped. If none remains, confidence is capped at 0.3 and a
``diagnosed`` status becomes ``possible_cause``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agent.schema import Diagnosis, Evidence, Status
from guardrails.preprocess import clean_line, split_raw_lines
from guardrails.redact import redact

MIN_SNIPPET_CHARS = 8
UNVERIFIED_CONFIDENCE_CAP = 0.3
CONTEXT_SOURCES = frozenset({"commit", "run_attempts"})

_WS = re.compile(r"\s+")
_ELISION = re.compile(r"\s*(?:\.\.\.|…)\s*")

FileReader = Callable[[str], "str | None"]  # path -> content, None if unavailable


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _matches(snippet: str, text: str) -> bool:
    """All elision-separated parts of ``snippet`` occur in ``text``, in order."""
    parts = [_norm(p) for p in _ELISION.split(snippet)]
    parts = [p for p in parts if p]
    if not parts or sum(len(p.replace(" ", "")) for p in parts) < MIN_SNIPPET_CHARS:
        return False
    haystack, pos = _norm(text), 0
    for part in parts:
        found = haystack.find(part, pos)
        if found < 0:
            return False
        pos = found + len(part)
    return True


def _snippet_lines(snippet: str) -> int:
    return max(1, len([s for s in snippet.strip().split("\n") if s.strip()]))


def _window(lines: Sequence[str], line: int, count: int) -> str | None:
    if line < 1 or line > len(lines):
        return None
    return "\n".join(lines[line - 1 : line - 1 + count])


class _Log:
    """The raw log viewed as the model saw it: cleaned + redacted, raw numbering."""

    def __init__(self, raw: str) -> None:
        self._lines = [redact(clean_line(t) or "") for t in split_raw_lines(raw)]

    def verify(self, ev: Evidence) -> bool:
        if ev.line is None:
            return False
        text = _window(self._lines, ev.line, _snippet_lines(ev.snippet))
        return text is not None and _matches(ev.snippet, text)


def _verify_file(ev: Evidence, read_file: FileReader) -> bool:
    content = read_file(ev.source)
    if content is None:
        return False
    if ev.line is None:
        return _matches(ev.snippet, content)
    text = _window(content.split("\n"), ev.line, _snippet_lines(ev.snippet))
    return text is not None and _matches(ev.snippet, text)


def _looks_like_path(source: str) -> bool:
    return "/" in source or "." in source


@dataclass(frozen=True)
class VerificationResult:
    diagnosis: Diagnosis  # verified evidence only, confidence/status adjusted
    dropped: tuple[Evidence, ...]  # cited but not verifiable (hallucinated-evidence metric)

    @property
    def cited(self) -> int:
        return len(self.diagnosis.evidence) + len(self.dropped)


def verify_evidence(
    diagnosis: Diagnosis,
    *,
    log: str,
    read_file: FileReader,
    context: Mapping[str, str] | None = None,
) -> VerificationResult:
    """Return the diagnosis with only verified evidence (``verified=True``)."""
    raw_log = _Log(log)
    ctx = context or {}
    kept: list[Evidence] = []
    dropped: list[Evidence] = []
    for ev in diagnosis.evidence:
        ev = ev.model_copy(update={"verified": False})  # never trust the model's flag
        if ev.source == "log":
            ok = raw_log.verify(ev)
        elif ev.source in CONTEXT_SOURCES:
            ok = ev.source in ctx and _matches(ev.snippet, ctx[ev.source])
        elif _looks_like_path(ev.source):
            ok = _verify_file(ev, read_file)
        else:
            ok = False
        (kept if ok else dropped).append(ev.model_copy(update={"verified": ok}))

    update: dict[str, object] = {"evidence": kept}
    if not kept:
        update["confidence"] = min(diagnosis.confidence, UNVERIFIED_CONFIDENCE_CAP)
        if diagnosis.status is Status.diagnosed:
            update["status"] = Status.possible_cause
    return VerificationResult(diagnosis=diagnosis.model_copy(update=update), dropped=tuple(dropped))
