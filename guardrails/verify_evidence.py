"""Evidence verifier: code, not the model, decides whether evidence is real.

The **snippet is the evidence; the line is only a hint.** A snippet matches when its
whitespace-normalized text occurs literally (no elisions, no paraphrase) in the
whitespace-normalized source text; a multi-line snippet must match consecutive lines.

* ``kind="log"``: the log being analyzed (``log``, if given, must name it). Lines are
  numbered as in the raw file and normalized exactly like the text the model saw
  (``preprocess.clean_log``, then ``redact.redact_log_lines``).
* ``kind="file"``: ``path`` at ``ref`` (default: ``merge`` for pull_request builds,
  ``head`` otherwise), read through the injected ``read_file``.
* ``kind="commit"``: a commit block the agent was shown (``sha`` selects one).
* ``kind="run_attempts"``: the run-attempts block the agent was shown.

Locating, for logs and files:

* ``line`` given: the snippet must match at ``line`` or within ±2 lines (nearest
  first); the offset is recorded. Further away: rejected.
* ``line`` missing: the snippet is located: its first occurrence (in a log, the first
  occurrence at or after the first fatal error, else the first anywhere).
* Not found: rejected.

``located_line`` is set on accepted log and file evidence; ``line`` keeps the model's
hint. Snippets shorter than ``MIN_SNIPPET_CHARS`` non-space characters never verify.
Recalled memories are hints, never evidence (there is no ``memory`` kind).

Unverified evidence is dropped. If none remains, confidence is capped at 0.3 and a
``diagnosed`` status becomes ``possible_cause``. ``fix_diff`` is checked separately
(``guardrails.verify_fix``). Every decision carries a human-readable reason.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agent.schema import Diagnosis, Evidence, Ref, Status
from guardrails.preprocess import clean_log, find_first_fatal
from guardrails.redact import redact_log_lines

MIN_SNIPPET_CHARS = 8
UNVERIFIED_CONFIDENCE_CAP = 0.3
MAX_LINE_OFFSET = 2
HINT_OFFSETS = (0, -1, 1, -2, 2)  # nearest first
COMMIT_PREFIX = "commit:"  # context key of one commit's block: "commit:<full sha>"
_SHOW = 160  # chars of a line quoted in a rejection reason
_SHA_PREFIX = re.compile(r"^[0-9a-f]{7,40}$")
_WS = re.compile(r"\s+")

# (path, ref) -> file content at that ref, or None if unavailable.
FileReader = Callable[[str, Ref], "str | None"]


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _too_short(snippet: str) -> bool:
    return len(_norm(snippet).replace(" ", "")) < MIN_SNIPPET_CHARS


def _exact(snippet: str, text: str) -> bool:
    return _norm(snippet) in _norm(text)


def _snippet_lines(snippet: str) -> int:
    return max(1, len([s for s in snippet.strip().split("\n") if s.strip()]))


def _window(lines: Sequence[str], line: int, count: int) -> str | None:
    if line < 1 or line > len(lines):
        return None
    return "\n".join(lines[line - 1 : line - 1 + count])


def _quote(text: str) -> str:
    text = _norm(text)
    return repr(text if len(text) <= _SHOW else text[:_SHOW] + "…")


def _occurrences(snippet: str, lines: Sequence[str], limit: int = 3) -> list[int]:
    count = _snippet_lines(snippet)
    found = []
    for n in range(1, len(lines) + 1):
        w = _window(lines, n, count)
        if w is not None and _exact(snippet, w):
            found.append(n)
            if len(found) == limit:
                break
    return found


@dataclass(frozen=True)
class Located:
    line: int | None  # where the snippet is; None: not found
    offset: int | None  # line - hint, when a hint was given and matched


def locate(snippet: str, lines: Sequence[str], hint: int | None, start: int = 1) -> Located:
    """Find ``snippet`` near ``hint`` (±2), or without a hint its first occurrence at or
    after ``start`` (wrapping to the top)."""
    count = _snippet_lines(snippet)
    if hint is not None:
        for offset in HINT_OFFSETS:
            w = _window(lines, hint + offset, count)
            if w is not None and _exact(snippet, w):
                return Located(hint + offset, offset)
        return Located(None, None)
    start = min(max(1, start), max(1, len(lines)))
    for n in [*range(start, len(lines) + 1), *range(1, start)]:
        w = _window(lines, n, count)
        if w is not None and _exact(snippet, w):
            return Located(n, None)
    return Located(None, None)


def _describe(
    where: str, hint: int | None, loc: Located, lines: Sequence[str], snippet: str
) -> str:
    if loc.line is not None:
        if hint is None:
            return f"located at {where} line {loc.line} (no line given)"
        if loc.offset == 0:
            return f"matches {where} line {loc.line}"
        return f"matches {where} line {loc.line} (cited {hint}, offset {loc.offset:+d})"
    elsewhere = _occurrences(snippet, lines)
    if hint is None:
        return f"snippet not found in {where}"
    at = _window(lines, hint, _snippet_lines(snippet))
    reads = f"line {hint} reads {_quote(at)}" if at is not None else f"no line {hint}"
    hint_text = (
        f"; it appears at line(s) {', '.join(f'{n} (offset {n - hint:+d})' for n in elsewhere)}"
        if elsewhere
        else "; it appears nowhere"
    )
    return f"not within ±{MAX_LINE_OFFSET} lines of {where} {reads}{hint_text}"


@dataclass(frozen=True)
class EvidenceCheck:
    # As cited, with ``verified`` and (for logs/files) ``located_line`` set by this
    # check, and a file's ``ref`` resolved to the version it was checked against.
    evidence: Evidence
    ok: bool
    reason: str
    offset: int | None = None  # located_line - line, when a line hint was given


class LogView:
    """The raw log as the model saw it: index ``n - 1`` holds raw line ``n``, cleaned and
    redacted with the same functions the pipeline uses; lines that cleaning removes
    (e.g. ``##[endgroup]``) are empty."""

    def __init__(self, raw: str) -> None:
        cleaned, total = clean_log(raw)
        redacted, _ = redact_log_lines(cleaned)
        lines = [""] * total
        for line in redacted:
            lines[line.number - 1] = line.text
        self.lines: list[str] = lines
        first = find_first_fatal(cleaned)
        self.first_fatal: int = first.line.number if first else 1


def _check_log(ev: Evidence, view: LogView, log_name: str | None) -> EvidenceCheck:
    if ev.log and log_name and ev.log != log_name:
        return EvidenceCheck(ev, False, f"cites log {ev.log!r}; this diagnosis is for {log_name!r}")
    loc = locate(ev.snippet, view.lines, ev.line, start=view.first_fatal)
    reason = _describe("log", ev.line, loc, view.lines, ev.snippet)
    if loc.line is None:
        return EvidenceCheck(ev, False, reason)
    checked = ev.model_copy(update={"verified": True, "located_line": loc.line})
    return EvidenceCheck(checked, True, reason, loc.offset)


def _check_file(ev: Evidence, read_file: FileReader, default_ref: Ref) -> EvidenceCheck:
    path = ev.path
    if not path:
        return EvidenceCheck(ev, False, "file evidence needs a path")
    ref: Ref = ev.ref or default_ref
    ev = ev.model_copy(update={"ref": ref})
    content = read_file(path, ref)
    if content is None:
        return EvidenceCheck(ev, False, f"{path} is not readable at {ref}")
    lines = content.split("\n")
    where = f"{path}@{ref}"
    loc = locate(ev.snippet, lines, ev.line)
    reason = _describe(where, ev.line, loc, lines, ev.snippet)
    if loc.line is None:
        return EvidenceCheck(ev, False, reason)
    checked = ev.model_copy(update={"verified": True, "located_line": loc.line})
    return EvidenceCheck(checked, True, reason, loc.offset)


def _check_block(ev: Evidence, context: Mapping[str, str]) -> EvidenceCheck:
    if ev.kind == "run_attempts":
        blocks = {"run_attempts": context.get("run_attempts")}
        label = "the run_attempts block"
    elif ev.sha:
        sha = ev.sha.strip().lower()
        if not _SHA_PREFIX.match(sha):
            return EvidenceCheck(ev, False, f"sha {ev.sha!r} is not a commit SHA (7-40 hex)")
        blocks = {
            k: v
            for k, v in context.items()
            if k.startswith(COMMIT_PREFIX) and k[len(COMMIT_PREFIX) :].startswith(sha)
        }
        label = f"commit {sha[:12]}"
    else:
        blocks = {"commit": context.get("commit")}
        label = "the commits shown"
    texts = [t for t in blocks.values() if t]
    if not texts:
        return EvidenceCheck(ev, False, f"the agent was never shown {label}")
    if any(_exact(ev.snippet, t) for t in texts):
        return EvidenceCheck(ev.model_copy(update={"verified": True}), True, f"found in {label}")
    return EvidenceCheck(ev, False, f"not found in {label}")


def check_evidence(
    ev: Evidence,
    *,
    log: LogView,
    read_file: FileReader,
    context: Mapping[str, str],
    default_ref: Ref = "head",
    log_name: str | None = None,
) -> EvidenceCheck:
    ev = ev.model_copy(update={"verified": False, "located_line": None})  # never trust these
    if _too_short(ev.snippet):
        return EvidenceCheck(ev, False, f"snippet shorter than {MIN_SNIPPET_CHARS} characters")
    if ev.kind == "log":
        return _check_log(ev, log, log_name)
    if ev.kind == "file":
        return _check_file(ev, read_file, default_ref)
    return _check_block(ev, context)


@dataclass(frozen=True)
class VerificationResult:
    diagnosis: Diagnosis  # verified evidence only, confidence/status adjusted
    checks: tuple[EvidenceCheck, ...]  # one per cited item, in citation order

    @property
    def dropped(self) -> tuple[Evidence, ...]:
        """Cited but not verifiable (hallucinated-evidence metric)."""
        return tuple(c.evidence for c in self.checks if not c.ok)

    @property
    def cited(self) -> int:
        return len(self.checks)


def verify_evidence(
    diagnosis: Diagnosis,
    *,
    log: str,
    read_file: FileReader,
    context: Mapping[str, str] | None = None,
    default_ref: Ref = "head",
    log_name: str | None = None,
) -> VerificationResult:
    """Return the diagnosis with only verified evidence (``verified=True``)."""
    view = LogView(log)
    checks = tuple(
        check_evidence(
            ev,
            log=view,
            read_file=read_file,
            context=context or {},
            default_ref=default_ref,
            log_name=log_name,
        )
        for ev in diagnosis.evidence
    )
    kept = [c.evidence for c in checks if c.ok]
    update: dict[str, object] = {"evidence": kept}
    if not kept:
        update["confidence"] = min(diagnosis.confidence, UNVERIFIED_CONFIDENCE_CAP)
        if diagnosis.status is Status.diagnosed:
            update["status"] = Status.possible_cause
    return VerificationResult(diagnosis=diagnosis.model_copy(update=update), checks=checks)


def default_ref_for(event: str) -> Ref:
    """The version a build compiled: the merge for pull_request builds, else head."""
    return "merge" if event == "pull_request" else "head"
