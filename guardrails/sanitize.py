"""G2 output sanitization, applied to every diagnosis before any notification.

Prose fields (``root_cause``, ``local_vs_ci``, evidence snippets):

* HTML stripped: comments (incl. spoofed ``<!-- cfa:... -->`` dedupe markers),
  ``<script>``/``<style>`` blocks with their content, and known HTML tags. Java
  generics such as ``List<String>`` are not HTML and are kept.
* ``@mentions`` neutralized by wrapping them in a code span (``@Override`` stays
  readable but pings nobody). Text already inside code spans is left alone, since
  GitHub does not resolve mentions there.
* URLs removed unless they point into the repo's own GitHub URLs; markdown links
  to other hosts keep only their text.
* Bidi overrides and control characters removed; length capped.

``fix_diff`` is code: it is only ever rendered inside a fenced block
(``render_code_block`` picks a fence longer than any backtick run in it, so it
cannot break out), so HTML and ``@`` stay as they are (Thymeleaf templates, Java
annotations). Non-allowlisted URLs are still removed and length is capped.

Secrets are redacted again (G3) as defense in depth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.schema import Diagnosis
from guardrails.redact import redact

LINK_REMOVED = "[link removed]"
TRUNCATED = " … [truncated]"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")

_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.S)
_HTML_BLOCK = re.compile(
    r"<(script|style|iframe|object|embed)\b[^>]*>.*?(?:</\1\s*>|$)", re.S | re.I
)
_HTML_TAGS = (
    "a|abbr|b|blockquote|body|br|button|code|dd|del|details|div|dl|dt|em|font|form|"
    "h[1-6]|head|hr|html|i|img|input|ins|kbd|li|link|meta|ol|p|picture|pre|q|s|samp|"
    "select|small|source|span|strike|strong|sub|summary|sup|svg|table|tbody|td|"
    "textarea|tfoot|th|thead|tr|tt|u|ul|var|video|audio|base|frame|frameset|marquee"
)
_HTML_TAG = re.compile(rf"</?(?:{_HTML_TAGS})\b(?:\s[^<>]*)?/?>", re.I)

_MD_LINK = re.compile(r"(!?)\[([^\]\n]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"\n]*\")?\s*\)")
_URL = re.compile(
    r"(?i)\b(?:javascript|vbscript|data):\S+"  # script-ish schemes: up to whitespace
    r"|\b(?:(?:https?|ftp):|www\.)[^\s<>()\[\]{}`'\"]+"
)
_TRAILING_PUNCT = ".,;:!?"

_CODE = re.compile(r"(```.*?(?:```|$)|`[^`\n]+`)", re.S)
_MENTION = re.compile(r"(?<![\w@`/.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})(?:/[A-Za-z0-9_.-]+)?")


@dataclass(frozen=True)
class SanitizeLimits:
    root_cause: int = 1_000
    local_vs_ci: int = 1_000
    snippet: int = 500
    fix_diff: int = 6_000


class _Urls:
    def __init__(self, repo: str) -> None:
        self._allowed = f"https://github.com/{repo.strip('/').lower()}"

    def allowed(self, url: str) -> bool:
        u = url.lower()
        return (
            u == self._allowed
            or u.startswith(self._allowed + "/")
            or u.startswith(self._allowed + "#")
        )

    def _bare(self, m: re.Match[str]) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in _TRAILING_PUNCT:
            url, trail = url[:-1], url[-1] + trail
        return (url if self.allowed(url) else LINK_REMOVED) + trail

    def _markdown(self, m: re.Match[str]) -> str:
        image, text, url = m.group(1), m.group(2), m.group(3)
        if self.allowed(url) and not image:
            return m.group(0)
        return text

    def scrub(self, text: str) -> str:
        text = _MD_LINK.sub(self._markdown, text)
        return _URL.sub(self._bare, text)


def strip_html(text: str) -> str:
    text = _HTML_COMMENT.sub("", text)
    text = _HTML_BLOCK.sub("", text)
    return _HTML_TAG.sub("", text)


def neutralize_mentions(text: str) -> str:
    parts = _CODE.split(text)
    for i in range(0, len(parts), 2):  # even indices: outside code
        parts[i] = _MENTION.sub(lambda m: f"`{m.group(0)}`", parts[i])
    return "".join(parts)


def cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATED))].rstrip() + TRUNCATED


def sanitize_text(text: str, *, repo: str, limit: int) -> str:
    text = _CONTROL.sub("", redact(text))
    text = strip_html(text)
    text = _Urls(repo).scrub(text)
    text = neutralize_mentions(text)
    return cap(text.strip(), limit)


def sanitize_code(text: str, *, repo: str, limit: int) -> str:
    text = _CONTROL.sub("", redact(text))
    text = _Urls(repo).scrub(text)
    if len(text) > limit:  # cut at a line boundary so the diff stays readable
        head = text[: max(0, limit - len(TRUNCATED))]
        text = head[: head.rfind("\n") + 1] if "\n" in head else head
        text += TRUNCATED.strip() + "\n"
    return text


def render_code_block(code: str, lang: str = "diff") -> str:
    """Fence ``code`` so nothing inside can close the block."""
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{code.rstrip(chr(10))}\n{fence}"


def sanitize_diagnosis(
    d: Diagnosis, *, repo: str, limits: SanitizeLimits | None = None
) -> Diagnosis:
    lim = limits or SanitizeLimits()
    return d.model_copy(
        update={
            "root_cause": sanitize_text(d.root_cause, repo=repo, limit=lim.root_cause),
            "local_vs_ci": sanitize_text(d.local_vs_ci, repo=repo, limit=lim.local_vs_ci),
            "evidence": [
                ev.model_copy(
                    update={
                        "snippet": sanitize_text(ev.snippet, repo=repo, limit=lim.snippet),
                        "source": sanitize_text(ev.source, repo=repo, limit=200),
                    }
                )
                for ev in d.evidence
            ],
            "fix_diff": (
                sanitize_code(d.fix_diff, repo=repo, limit=lim.fix_diff) if d.fix_diff else None
            ),
        }
    )
