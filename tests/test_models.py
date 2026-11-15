"""Model provider config, pinning, clear errors, ping accounting. No network unless
CFA_LIVE_MODEL=1 (then one real call to the configured model)."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import ollama
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import cli
import telemetry
from agent import models
from agent.models import (
    ModelConfigError,
    ModelHandle,
    ModelSettings,
    ModelUnavailableError,
    build_model,
    check_pinned,
    ping,
)
from guardrails.budget import Price

QWEN = "qwen2.5:7b-instruct"
QWEN_DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
OLLAMA_ENV = {"MODEL_PROVIDER": "ollama", "TRIAGE_MODEL_ID": QWEN}

# ----------------------------------------------------------------- settings


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "MODEL_PROVIDER is not set"),
        ({"MODEL_PROVIDER": "openai"}, "not supported"),
        ({"MODEL_PROVIDER": "ollama", "MODEL_TIMEOUT_S": "soon"}, "MODEL_TIMEOUT_S"),
        ({"MODEL_PROVIDER": "ollama", "MODEL_TIMEOUT_S": "-1"}, "positive"),
        ({"MODEL_PROVIDER": "bedrock", "MODEL_PRICE_INPUT_PER_MTOK": "3"}, "both"),
        (
            {
                "MODEL_PROVIDER": "bedrock",
                "MODEL_PRICE_INPUT_PER_MTOK": "x",
                "MODEL_PRICE_OUTPUT_PER_MTOK": "1",
            },
            "not a number",
        ),
    ],
)
def test_invalid_env_is_a_clear_config_error(env: dict[str, str], message: str) -> None:
    with pytest.raises(ModelConfigError, match=message):
        ModelSettings.from_env(env)


def test_settings_from_env() -> None:
    s = ModelSettings.from_env(
        {**OLLAMA_ENV, "MODEL_PROVIDER": " Ollama ", "OLLAMA_NUM_CTX": "8192"}
    )
    assert s.provider == "ollama"
    assert s.model_id("triage") == QWEN
    assert s.ollama_num_ctx == 8192
    assert s.price == Price(0.0, 0.0)  # local models are free
    with pytest.raises(ModelConfigError, match="SPECIALIST_MODEL_ID is not set"):
        s.model_id("specialist")


def test_cloud_price_is_configured_or_unknown() -> None:
    assert ModelSettings.from_env({"MODEL_PROVIDER": "bedrock"}).price is None
    s = ModelSettings.from_env(
        {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_PRICE_INPUT_PER_MTOK": "3",
            "MODEL_PRICE_OUTPUT_PER_MTOK": "15",
        }
    )
    assert s.price == Price(3.0, 15.0)


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("ollama", QWEN),
        ("ollama", f"{QWEN}@845dbda0"),
        ("ollama", "llama3.1:latest@46e0c10c"),  # latest is fine when the digest pins it
        ("ollama", "library/qwen2.5-coder:32b-instruct-q4_K_M"),
        ("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"),
        ("bedrock", "us.meta.llama3-1-8b-instruct-v1:0"),
        ("anthropic", "claude-sonnet-5-5"),
    ],
)
def test_pinned_ids_accepted(provider: models.Provider, model_id: str) -> None:
    check_pinned(provider, model_id)


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("ollama", "qwen2.5"),  # no tag = latest
        ("ollama", "llama3.1:latest"),
        ("ollama", "qwen2.5:7b@xyz"),
        ("bedrock", "anthropic.claude-3-haiku-20240307-v1"),
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("anthropic", "gpt-4o"),
    ],
)
def test_floating_ids_rejected(provider: models.Provider, model_id: str) -> None:
    with pytest.raises(ModelConfigError):
        check_pinned(provider, model_id)


# ------------------------------------------------------------- build: ollama


@dataclass
class FakeOllamaClient:
    installed: dict[str, str] = field(default_factory=lambda: {QWEN: QWEN_DIGEST})
    error: Exception | None = None

    def __call__(self, **_: Any) -> FakeOllamaClient:  # stands in for ollama.Client(...)
        return self

    def list(self) -> SimpleNamespace:
        if self.error:
            raise self.error
        return SimpleNamespace(
            models=[SimpleNamespace(model=m, digest=d) for m, d in self.installed.items()]
        )


def build_ollama(
    monkeypatch: pytest.MonkeyPatch, client: FakeOllamaClient, **env: str
) -> ModelHandle:
    monkeypatch.setattr(ollama, "Client", client)
    return build_model(ModelSettings.from_env({**OLLAMA_ENV, **env}), "triage")


def test_ollama_records_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = build_ollama(monkeypatch, FakeOllamaClient())
    assert handle.pinned_id == f"{QWEN}@845dbda0ea48"
    assert handle.price == Price(0.0, 0.0)
    cfg = handle.model.get_config()
    assert cfg["model_id"] == QWEN  # the digest suffix is ours, not Ollama's
    assert cfg["options"] == {"num_ctx": 16384}
    assert cfg["temperature"] == 0.0
    assert cfg["max_tokens"] == 4096  # bounds every call, incl. the constrained fallback
    capped = build_ollama(monkeypatch, FakeOllamaClient(), MODEL_MAX_OUTPUT_TOKENS="1024")
    assert capped.model.get_config()["max_tokens"] == 1024


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), ConnectionError("Failed to connect to Ollama")],
)
def test_ollama_unreachable(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    with pytest.raises(ModelUnavailableError, match=r"cannot reach Ollama.*ollama serve"):
        build_ollama(monkeypatch, FakeOllamaClient(error=error))


def test_ollama_model_not_pulled(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ModelUnavailableError, match=r"ollama pull qwen2\.5:14b"):
        build_ollama(monkeypatch, FakeOllamaClient(), TRIAGE_MODEL_ID="qwen2.5:14b")


def test_ollama_digest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ModelConfigError, match="pins digest deadbeef"):
        build_ollama(monkeypatch, FakeOllamaClient(), TRIAGE_MODEL_ID=f"{QWEN}@deadbeef")


# ------------------------------------------------------ build: cloud providers


def test_anthropic_requires_key() -> None:
    s = ModelSettings.from_env({"MODEL_PROVIDER": "anthropic", "TRIAGE_MODEL_ID": "claude-x-1"})
    with pytest.raises(ModelConfigError, match="ANTHROPIC_API_KEY is not set"):
        build_model(s, "triage")


def test_anthropic_sdk_missing_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    s = ModelSettings.from_env(
        {"MODEL_PROVIDER": "anthropic", "TRIAGE_MODEL_ID": "claude-x-1", "ANTHROPIC_API_KEY": "k"}
    )
    with pytest.raises(ModelConfigError, match="uv sync --extra anthropic"):
        build_model(s, "triage")


def test_bedrock_requires_region_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"MODEL_PROVIDER": "bedrock", "TRIAGE_MODEL_ID": "meta.llama3-8b-instruct-v1:0"}
    with pytest.raises(ModelConfigError, match="AWS_REGION is not set"):
        build_model(ModelSettings.from_env(env), "triage")

    import boto3

    monkeypatch.setattr(boto3.Session, "get_credentials", lambda self: None)
    with pytest.raises(ModelConfigError, match="no AWS credentials"):
        build_model(ModelSettings.from_env({**env, "AWS_REGION": "us-east-1"}), "triage")


# ---------------------------------------------------------------------- ping


class FakeAgent:
    error: Exception | None = None

    def __init__(self, **_: Any) -> None:
        pass

    def __call__(self, prompt: str) -> FakeResult:
        if self.error:
            raise self.error
        return FakeResult(
            SimpleNamespace(accumulated_usage={"inputTokens": 1_000, "outputTokens": 200})
        )


@dataclass
class FakeResult:
    metrics: SimpleNamespace

    def __str__(self) -> str:
        return "pong\n"


def handle(price: Price | None) -> ModelHandle:
    return ModelHandle("bedrock", "triage", "m:0", "m:0", price, model=object())  # type: ignore[arg-type]


def test_ping_logs_tokens_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "Agent", FakeAgent)
    result = ping(handle(Price(3.0, 15.0)))
    assert (result.reply, result.input_tokens, result.output_tokens) == ("pong", 1_000, 200)
    assert result.cost_usd == pytest.approx(0.003 + 0.003)
    assert ping(handle(None)).cost_usd is None


def test_ping_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing(FakeAgent):
        error = RuntimeError("401 invalid x-api-key")

    monkeypatch.setattr(models, "Agent", Failing)
    with pytest.raises(ModelUnavailableError, match="RuntimeError: 401 invalid x-api-key"):
        ping(handle(None))


# ----------------------------------------------------------------------- CLI


def test_cli_ping_model_config_error_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    assert cli.main(["ping-model"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "error: MODEL_PROVIDER is not set" in err
    assert "Traceback" not in err


def test_cli_ping_model_unavailable_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for k, v in OLLAMA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(ollama, "Client", FakeOllamaClient(error=httpx.ConnectError("refused")))
    assert cli.main(["ping-model"]) == 1
    _, err = capsys.readouterr()
    assert "error: cannot reach Ollama" in err and "Traceback" not in err


def test_cli_ping_model_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for k, v in OLLAMA_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(ollama, "Client", FakeOllamaClient())
    monkeypatch.setattr(models, "Agent", FakeAgent)
    assert cli.main(["ping-model"]) == 0
    out, err = capsys.readouterr()
    data = json.loads(out)
    assert data["pinned_id"] == f"{QWEN}@845dbda0ea48"
    assert (data["input_tokens"], data["output_tokens"], data["cost_usd"]) == (1_000, 200, 0.0)
    assert "[span] model.ping" in err and "input_tokens=1000" in err


# ----------------------------------------------------------------- telemetry


def test_exporter_strips_event_content_but_keeps_exceptions() -> None:
    exporter = InMemorySpanExporter()
    tracer = telemetry.get_tracer(telemetry.make_provider(exporter))
    with tracer.start_as_current_span("chat") as span:
        span.add_event("gen_ai.user.message", {"content": "log text AKIA..."})
        span.record_exception(ValueError("boom"))
        span.set_status(StatusCode.ERROR)
    (exported,) = exporter.get_finished_spans()
    by_name = {e.name: dict(e.attributes or {}) for e in exported.events}
    assert by_name["gen_ai.user.message"] == {}
    assert by_name["exception"]["exception.type"] == "ValueError"


# ---------------------------------------------------------------------- live


@pytest.mark.skipif(os.environ.get("CFA_LIVE_MODEL") != "1", reason="set CFA_LIVE_MODEL=1")
def test_live_ping() -> None:
    h = build_model(ModelSettings.from_env(), "triage")
    result = ping(h)
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert result.cost_usd is not None or h.provider != "ollama"
