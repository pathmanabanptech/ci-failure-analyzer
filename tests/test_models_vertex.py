"""Claude on Vertex AI provider, and Claude-wide settings (no sampling params, output cap)."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest

from agent import models
from agent.models import ModelConfigError, ModelSettings, build_model, check_pinned

VERTEX_ENV = {
    "MODEL_PROVIDER": "vertex",
    "SPECIALIST_MODEL_ID": "claude-opus-5-5",
    "VERTEX_PROJECT_ID": "cfa-dev",
}


def _fake_adc(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda *a, **k: (SimpleNamespace(), "cfa-dev"))


# ------------------------------------------------------------------ settings


def test_vertex_settings_defaults_and_fallbacks() -> None:
    s = ModelSettings.from_env(VERTEX_ENV)
    assert (s.provider, s.vertex_project, s.vertex_region) == ("vertex", "cfa-dev", "global")
    env = {k: v for k, v in VERTEX_ENV.items() if k != "VERTEX_PROJECT_ID"}
    s = ModelSettings.from_env(
        {**env, "GOOGLE_CLOUD_PROJECT": "other", "VERTEX_REGION": "us-east5"}
    )
    assert (s.vertex_project, s.vertex_region) == ("other", "us-east5")


@pytest.mark.parametrize(
    ("provider", "expected"), [("vertex", 16_000), ("anthropic", 16_000), ("ollama", 4096)]
)
def test_claude_providers_default_to_16k_output(provider: str, expected: int) -> None:
    assert ModelSettings.from_env({"MODEL_PROVIDER": provider}).max_output_tokens == expected
    env = {"MODEL_PROVIDER": provider, "MODEL_MAX_OUTPUT_TOKENS": "2000"}
    assert ModelSettings.from_env(env).max_output_tokens == 2000  # explicit setting wins


@pytest.mark.parametrize(
    "model_id", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-4-5@20251101"]
)
def test_vertex_pinned_ids(model_id: str) -> None:
    check_pinned("vertex", model_id)


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("vertex", "claude-sonnet-latest"),
        ("vertex", "gemini-2.5-pro"),
        ("vertex", "claude-opus-4-5@2025"),
        ("anthropic", "claude-opus-4-5@20251101"),  # snapshot suffix is Vertex-only
    ],
)
def test_vertex_floating_or_foreign_ids_rejected(provider: models.Provider, model_id: str) -> None:
    with pytest.raises(ModelConfigError):
        check_pinned(provider, model_id)


# -------------------------------------------------------------------- vertex


def test_vertex_requires_project() -> None:
    env = {k: v for k, v in VERTEX_ENV.items() if k != "VERTEX_PROJECT_ID"}
    with pytest.raises(ModelConfigError, match="VERTEX_PROJECT_ID"):
        build_model(ModelSettings.from_env(env), "specialist")


def test_vertex_sdk_missing_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModelConfigError, match="uv sync --extra vertex"):
        build_model(ModelSettings.from_env(VERTEX_ENV), "specialist")


def test_vertex_without_google_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("google.auth")
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_adc(*a: Any, **k: Any) -> Any:
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(ModelConfigError, match="gcloud auth application-default login"):
        build_model(ModelSettings.from_env(VERTEX_ENV), "specialist")


def test_vertex_builds_claude_with_vertex_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    pytest.importorskip("google.auth")
    from anthropic import AsyncAnthropicVertex

    _fake_adc(monkeypatch)
    env = {
        **VERTEX_ENV,
        "MODEL_PRICE_INPUT_PER_MTOK": "4",
        "MODEL_PRICE_OUTPUT_PER_MTOK": "20",
        "MODEL_TIMEOUT_S": "300",
    }
    handle = build_model(ModelSettings.from_env(env), "specialist")
    assert handle.provider == "vertex" and handle.pinned_id == "claude-opus-5-5"
    client = handle.model.client  # type: ignore[attr-defined]
    assert isinstance(client, AsyncAnthropicVertex)
    assert (client.project_id, client.region, client.max_retries) == ("cfa-dev", "global", 0)
    config = handle.model.get_config()
    assert config["model_id"] == "claude-opus-5-5" and config["max_tokens"] == 16_000
    assert "params" not in config  # no temperature: current Claude models reject it
    assert handle.price is not None and handle.price.cost(1_000_000, 0) == 4.0


# -------------------------------------------- sampling params on other providers


def test_anthropic_sends_no_temperature() -> None:
    pytest.importorskip("anthropic")
    env = {
        "MODEL_PROVIDER": "anthropic",
        "SPECIALIST_MODEL_ID": "claude-opus-5-5",
        "ANTHROPIC_API_KEY": "test-key",
    }
    config = build_model(ModelSettings.from_env(env), "specialist").model.get_config()
    assert "params" not in config and config["max_tokens"] == 16_000


@pytest.mark.parametrize(
    ("model_id", "temperature"),
    [("anthropic.claude-opus-5-5-v1:0", None), ("amazon.nova-pro-v1:0", 0.0)],
)
def test_bedrock_sends_temperature_only_to_non_claude(
    monkeypatch: pytest.MonkeyPatch, model_id: str, temperature: float | None
) -> None:
    import boto3
    import strands.models.bedrock as bedrock_mod

    seen: dict[str, Any] = {}

    class FakeBedrock:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(boto3.Session, "get_credentials", lambda self: object())
    monkeypatch.setattr(bedrock_mod, "BedrockModel", FakeBedrock)
    env = {"MODEL_PROVIDER": "bedrock", "SPECIALIST_MODEL_ID": model_id, "AWS_REGION": "us-east-1"}
    build_model(ModelSettings.from_env(env), "specialist")
    assert seen.get("temperature") == temperature
