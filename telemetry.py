"""OpenTelemetry setup. Every run emits one span per pipeline stage.

Only the exporter changes between environments: console locally (this module),
Jaeger/CloudWatch later. Span attributes carry counts, line numbers, kinds and
SHAs, never log text, file contents or diagnosis prose (those may hold secrets
or injected content).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Literal, TextIO

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
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
    return str(value)


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
        parts.append(f"event={event.name}({_fmt_attrs(event.attributes)})")
    return " ".join(parts) + "\n"


def console_exporter(stream: TextIO | None = None, fmt: SpanFormat = "compact") -> SpanExporter:
    """The OTel console exporter, on stderr by default so stdout stays machine-readable."""
    formatter: Callable[[ReadableSpan], str] = (
        compact_line if fmt == "compact" else lambda s: s.to_json() + "\n"
    )
    return ConsoleSpanExporter(out=stream or sys.stderr, formatter=formatter)


def make_provider(exporter: SpanExporter | None) -> TracerProvider:
    """A tracer provider exporting synchronously to ``exporter`` (None: spans are dropped)."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def configure(exporter: SpanExporter | None) -> TracerProvider:
    """Make the provider process-global (agent frameworks emit spans through it too)."""
    provider = make_provider(exporter)
    trace.set_tracer_provider(provider)
    return provider


def get_tracer(provider: trace.TracerProvider | None = None) -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME, tracer_provider=provider)
