"""OpenTelemetry setup. Every run emits one span per pipeline stage.

Only the exporter changes between environments: console locally (this module),
Jaeger/CloudWatch later. Span attributes carry counts, line numbers, kinds and
SHAs, never log text, file contents or diagnosis prose (those may hold secrets
or injected content). Agent frameworks put prompts and replies in span events;
the exporter strips event attributes before anything is written.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TextIO

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue

SERVICE_NAME = "ci-failure-analyzer"
TRACER_NAME = "cfa"

SpanFormat = Literal["compact", "json"]


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list | tuple):
        return ",".join(str(v) for v in value)
    return " ".join(str(value).split())  # one line per span, even for multi-line values


def _fmt_attrs(attrs: Mapping[str, AttributeValue] | None) -> str:
    return " ".join(f"{k}={_fmt_value(v)}" for k, v in (attrs or {}).items())


def compact_line(span: ReadableSpan) -> str:
    """``[span] <name> <ms>ms [ERROR] key=value ...`` (one line per span)."""
    ms = ((span.end_time or 0) - (span.start_time or 0)) / 1e6
    parts = [f"[span] {span.name}", f"{ms:.1f}ms"]
    if span.status.status_code is StatusCode.ERROR:
        parts.append("ERROR")
    if attrs := _fmt_attrs(span.attributes):
        parts.append(attrs)
    for event in span.events:  # e.g. recorded exceptions
        attrs = _fmt_attrs(event.attributes)
        parts.append(f"event={event.name}({attrs})" if attrs else f"event={event.name}")
    return " ".join(parts) + "\n"


def console_exporter(stream: TextIO | None = None, fmt: SpanFormat = "compact") -> SpanExporter:
    """The OTel console exporter, on stderr by default so stdout stays machine-readable."""
    formatter: Callable[[ReadableSpan], str] = (
        compact_line if fmt == "compact" else lambda s: s.to_json() + "\n"
    )
    return ConsoleSpanExporter(out=stream or sys.stderr, formatter=formatter)


class WithoutContent(SpanExporter):
    """Drops event attributes (agent frameworks record prompts, tool results and replies
    there) before export; event names stay. Exceptions recorded on spans are kept."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    @staticmethod
    def _strip(span: ReadableSpan) -> ReadableSpan:
        events = tuple(
            e if e.name == "exception" else Event(e.name, timestamp=e.timestamp)
            for e in span.events
        )
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=span.attributes,
            events=events,
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._strip(s) for s in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


def make_provider(exporter: SpanExporter | None) -> TracerProvider:
    """A tracer provider exporting synchronously to ``exporter`` (None: spans are dropped).

    Event content is always stripped (see ``WithoutContent``).
    """
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(WithoutContent(exporter)))
    return provider


def configure(exporter: SpanExporter | None) -> TracerProvider:
    """Make the provider process-global (agent frameworks emit spans through it too)."""
    provider = make_provider(exporter)
    trace.set_tracer_provider(provider)
    return provider


def get_tracer(provider: trace.TracerProvider | None = None) -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME, tracer_provider=provider)
