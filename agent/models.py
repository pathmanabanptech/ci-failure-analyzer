"""Model provider selection. The provider is configuration, never code.

Environment:

    MODEL_PROVIDER        ollama | bedrock | anthropic | vertex
    TRIAGE_MODEL_ID       exact model version for triage
    SPECIALIST_MODEL_ID   exact model version for the diagnosing agent(s)
    MODEL_TIMEOUT_S       per-request timeout (default 120)
    MODEL_MAX_OUTPUT_TOKENS  cap on tokens generated per model call (default 16000 for
                          Claude providers, whose thinking counts toward it; 4096
                          otherwise); also bounds a constrained-decoding call that would
                          otherwise loop
    MODEL_PRICE_INPUT_PER_MTOK / MODEL_PRICE_OUTPUT_PER_MTOK
                          USD per million tokens, for cost accounting (cloud providers;
                          a local Ollama model costs 0)

    OLLAMA_HOST           default http://localhost:11434
    OLLAMA_NUM_CTX        context window to request (default 16384); Ollama's own
                          default is small and silently truncates longer prompts
    AWS_REGION            Bedrock region (AWS credentials from the usual chain)
    ANTHROPIC_API_KEY     Anthropic API key
    VERTEX_PROJECT_ID     Google Cloud project for Claude on Vertex AI (falls back to
                          GOOGLE_CLOUD_PROJECT); credentials from Application Default
                          Credentials (`gcloud auth application-default login`, or the
                          attached service account on Google Cloud)
    VERTEX_REGION         Vertex AI region (default ``global``)

Model IDs must be pinned to an exact version, because every eval result records them:

* Ollama: ``name:tag`` with an explicit tag, optionally ``name:tag@<digest prefix>``
  (checked against the server's digest); ``latest`` only with a digest. The resolved
  digest is always recorded in ``ModelHandle.pinned_id``.
* Bedrock: ends with a ``:<n>`` version suffix.
* Anthropic / Vertex: an exact Claude id, no ``-latest`` aliases (Vertex also takes an
  ``@YYYYMMDD`` snapshot suffix for older models).

Claude models are never sent sampling parameters (``temperature``): current models
reject them with a 400.

Errors are typed: ``ModelConfigError`` (env is wrong; fix the config) and
``ModelUnavailableError`` (provider unreachable, model missing, credentials rejected).
Neither carries a stack trace to the user.
"""

from __future__ import annotations

import importlib.util
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from strands import Agent
from strands.models import Model

from agent.errors import ModelConfigError, ModelError, ModelUnavailableError
from guardrails.budget import Price

__all__ = [
    "ModelConfigError",
    "ModelError",
    "ModelHandle",
    "ModelSettings",
    "ModelUnavailableError",
    "PingResult",
    "build_model",
    "check_pinned",
    "ping",
]

Provider = Literal["ollama", "bedrock", "anthropic", "vertex"]
Role = Literal["triage", "specialist"]

PROVIDERS: tuple[Provider, ...] = get_args(Provider)
ROLE_ENV: dict[Role, str] = {"triage": "TRIAGE_MODEL_ID", "specialist": "SPECIALIST_MODEL_ID"}
LOCAL_PRICE = Price(input_per_mtok=0.0, output_per_mtok=0.0)

_OLLAMA_ID = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9._/-]*):(?P<tag>[A-Za-z0-9._-]+)(?:@(?P<digest>[0-9a-f]{6,64}))?$"
)
_BEDROCK_ID = re.compile(r"^[A-Za-z0-9._-]+:\d+$")
_ANTHROPIC_ID = re.compile(r"^claude-[a-z0-9.-]+$")
_VERTEX_ID = re.compile(r"^claude-[a-z0-9.-]+(?:@\d{8})?$")
CLAUDE_PROVIDERS: frozenset[Provider] = frozenset({"anthropic", "vertex"})
CLAUDE_MAX_OUTPUT_TOKENS = 16_000
DEFAULT_MAX_OUTPUT_TOKENS = 4096


# ------------------------------------------------------------------- settings


@dataclass(frozen=True)
class ModelSettings:
    provider: Provider
    model_ids: Mapping[Role, str | None]
    timeout_s: float = 120.0
    max_output_tokens: int = 4096
    price: Price | None = None  # None: unknown (cloud provider without a configured price)
    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: int = 16_384
    aws_region: str | None = None
    anthropic_api_key: str | None = None
    vertex_project: str | None = None
    vertex_region: str = "global"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelSettings:
        env = os.environ if env is None else env
        provider = env.get("MODEL_PROVIDER", "").strip().lower()
        if not provider:
            raise ModelConfigError(f"MODEL_PROVIDER is not set (one of: {', '.join(PROVIDERS)})")
        if provider not in PROVIDERS:
            raise ModelConfigError(
                f"MODEL_PROVIDER={provider!r} is not supported (one of: {', '.join(PROVIDERS)})"
            )
        prov: Provider = provider
        return cls(
            provider=prov,
            model_ids={r: (env.get(var) or "").strip() or None for r, var in ROLE_ENV.items()},
            timeout_s=_positive_float(env, "MODEL_TIMEOUT_S", 120.0),
            max_output_tokens=int(
                _positive_float(
                    env,
                    "MODEL_MAX_OUTPUT_TOKENS",
                    CLAUDE_MAX_OUTPUT_TOKENS
                    if prov in CLAUDE_PROVIDERS
                    else DEFAULT_MAX_OUTPUT_TOKENS,
                )
            ),
            price=LOCAL_PRICE if prov == "ollama" else _price(env),
            ollama_host=env.get("OLLAMA_HOST") or "http://localhost:11434",
            ollama_num_ctx=int(_positive_float(env, "OLLAMA_NUM_CTX", 16_384)),
            aws_region=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or None,
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
            vertex_project=env.get("VERTEX_PROJECT_ID") or env.get("GOOGLE_CLOUD_PROJECT") or None,
            vertex_region=env.get("VERTEX_REGION") or "global",
        )

    def model_id(self, role: Role) -> str:
        value = self.model_ids.get(role)
        if not value:
            raise ModelConfigError(f"{ROLE_ENV[role]} is not set")
        check_pinned(self.provider, value, ROLE_ENV[role])
        return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ModelConfigError(f"{name}={raw!r} is not a number") from None
    if value <= 0:
        raise ModelConfigError(f"{name} must be positive, got {raw!r}")
    return value


def _price(env: Mapping[str, str]) -> Price | None:
    names = ("MODEL_PRICE_INPUT_PER_MTOK", "MODEL_PRICE_OUTPUT_PER_MTOK")
    present = [n for n in names if env.get(n)]
    if not present:
        return None
    if len(present) == 1:
        raise ModelConfigError(f"set both {names[0]} and {names[1]}, or neither")
    values = []
    for n in names:
        try:
            v = float(env[n])
        except ValueError:
            raise ModelConfigError(f"{n}={env[n]!r} is not a number") from None
        if v < 0:
            raise ModelConfigError(f"{n} must not be negative")
        values.append(v)
    return Price(input_per_mtok=values[0], output_per_mtok=values[1])


def check_pinned(provider: Provider, model_id: str, var: str = "model id") -> None:
    """Reject floating model IDs (no tag, ``latest`` aliases, no version suffix)."""
    if provider == "ollama":
        m = _OLLAMA_ID.match(model_id)
        if not m:
            raise ModelConfigError(
                f"{var}={model_id!r} must be 'name:tag' (e.g. qwen2.5:7b-instruct), "
                "optionally '@<digest prefix>'"
            )
        if m["tag"] == "latest" and not m["digest"]:
            raise ModelConfigError(
                f"{var}={model_id!r} is a floating tag; use an explicit tag "
                "(see `ollama list`) or append '@<digest prefix>'"
            )
    elif provider == "bedrock":
        if not _BEDROCK_ID.match(model_id):
            raise ModelConfigError(
                f"{var}={model_id!r} must be an exact Bedrock model id with a ':<n>' version"
            )
    else:
        pattern = _VERTEX_ID if provider == "vertex" else _ANTHROPIC_ID
        if not pattern.match(model_id) or model_id.endswith("-latest"):
            raise ModelConfigError(
                f"{var}={model_id!r} must be an exact Claude model id, no aliases"
            )


# --------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelHandle:
    provider: Provider
    role: Role
    model_id: str  # as configured
    pinned_id: str  # what eval results record (Ollama: name:tag@digest)
    price: Price | None
    model: Model


def build_model(settings: ModelSettings, role: Role) -> ModelHandle:
    """Validate config and reachability, then build the Strands model. No model call."""
    model_id = settings.model_id(role)
    if settings.provider == "ollama":
        return _ollama(settings, role, model_id)
    if settings.provider == "bedrock":
        return _bedrock(settings, role, model_id)
    if settings.provider == "vertex":
        return _vertex(settings, role, model_id)
    return _anthropic(settings, role, model_id)


def _ollama(settings: ModelSettings, role: Role, model_id: str) -> ModelHandle:
    import httpx
    import ollama
    from strands.models.ollama import OllamaModel

    m = _OLLAMA_ID.match(model_id)
    assert m is not None  # checked by model_id()
    name_tag, want_digest = model_id.split("@")[0], m["digest"]
    host = settings.ollama_host
    client = ollama.Client(host=host, timeout=min(settings.timeout_s, 10.0))
    try:
        installed = {x.model: x.digest for x in client.list().models if x.model}
    except (httpx.HTTPError, ollama.ResponseError, ConnectionError) as e:
        raise ModelUnavailableError(
            f"cannot reach Ollama at {host} ({e}); is `ollama serve` running?"
        ) from None
    digest = installed.get(name_tag)
    if digest is None:
        raise ModelUnavailableError(
            f"model {name_tag!r} is not pulled on {host}; run `ollama pull {name_tag}`"
        )
    if want_digest and not digest.startswith(want_digest):
        raise ModelConfigError(
            f"{ROLE_ENV[role]} pins digest {want_digest} but {name_tag} on {host} is "
            f"{digest[:12]}; re-pull the pinned version or update the pin"
        )
    model = OllamaModel(
        host,
        ollama_client_args={"timeout": settings.timeout_s},
        model_id=name_tag,
        temperature=0.0,
        max_tokens=settings.max_output_tokens,  # sent as num_predict, on every call
        options={"num_ctx": settings.ollama_num_ctx},
        keep_alive="10m",
    )
    return ModelHandle(
        provider="ollama",
        role=role,
        model_id=model_id,
        pinned_id=f"{name_tag}@{digest[:12]}",
        price=LOCAL_PRICE,
        model=model,
    )


def _bedrock(settings: ModelSettings, role: Role, model_id: str) -> ModelHandle:
    import boto3
    from botocore.config import Config
    from strands.models.bedrock import BedrockModel

    if not settings.aws_region:
        raise ModelConfigError("AWS_REGION is not set (required for MODEL_PROVIDER=bedrock)")
    session = boto3.Session(region_name=settings.aws_region)
    if session.get_credentials() is None:
        raise ModelConfigError(
            "no AWS credentials found (AWS_PROFILE, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
            "or `aws sso login`)"
        )
    model = BedrockModel(
        boto_session=session,
        boto_client_config=Config(
            read_timeout=settings.timeout_s, connect_timeout=10, retries={"max_attempts": 1}
        ),
        model_id=model_id,
        max_tokens=settings.max_output_tokens,
        # Claude on Bedrock rejects sampling parameters (current models); None is not sent.
        temperature=None if "claude" in model_id else 0.0,
    )
    return ModelHandle("bedrock", role, model_id, model_id, settings.price, model)


def _anthropic(settings: ModelSettings, role: Role, model_id: str) -> ModelHandle:
    if not settings.anthropic_api_key:
        raise ModelConfigError(
            "ANTHROPIC_API_KEY is not set (required for MODEL_PROVIDER=anthropic)"
        )
    if importlib.util.find_spec("anthropic") is None:
        raise ModelConfigError(
            "the Anthropic SDK is not installed; run `uv sync --extra anthropic`"
        )
    from strands.models.anthropic import AnthropicModel

    model = AnthropicModel(
        client_args={
            "api_key": settings.anthropic_api_key,
            "timeout": settings.timeout_s,
            "max_retries": 0,
        },
        model_id=model_id,
        max_tokens=settings.max_output_tokens,
    )
    return ModelHandle("anthropic", role, model_id, model_id, settings.price, model)


def _vertex(settings: ModelSettings, role: Role, model_id: str) -> ModelHandle:
    """Claude on Google Cloud Vertex AI (Google managed). Auth: Application Default
    Credentials; no API key. Checked locally here; the first call verifies access."""
    if not settings.vertex_project:
        raise ModelConfigError(
            "VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not set (required for "
            "MODEL_PROVIDER=vertex)"
        )
    if (
        importlib.util.find_spec("anthropic") is None
        or importlib.util.find_spec("google.auth") is None
    ):
        raise ModelConfigError(
            "the Anthropic Vertex SDK is not installed; run `uv sync --extra vertex`"
        )
    import google.auth
    from anthropic import AsyncAnthropicVertex
    from google.auth.exceptions import DefaultCredentialsError
    from strands.models.anthropic import AnthropicModel

    try:
        google.auth.default()
    except DefaultCredentialsError:
        raise ModelConfigError(
            "no Google credentials found; run `gcloud auth application-default login` "
            "(on Google Cloud the attached service account is used)"
        ) from None
    client = AsyncAnthropicVertex(
        project_id=settings.vertex_project,
        region=settings.vertex_region,
        timeout=settings.timeout_s,
        max_retries=0,
    )
    model = AnthropicModel(client=client, model_id=model_id, max_tokens=settings.max_output_tokens)
    return ModelHandle("vertex", role, model_id, model_id, settings.price, model)


# ----------------------------------------------------------------------- ping

PING_SYSTEM_PROMPT = "You are a health check. Reply with exactly one word: pong"


@dataclass(frozen=True)
class PingResult:
    reply: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float | None  # None when the provider's price is not configured


def ping(handle: ModelHandle) -> PingResult:
    """One minimal model call. Any failure becomes ModelUnavailableError (no retry)."""
    agent = Agent(model=handle.model, system_prompt=PING_SYSTEM_PROMPT, callback_handler=None)
    start = time.monotonic()
    try:
        result = agent("ping")
    except Exception as e:  # provider SDKs raise many unrelated types; the message is enough
        raise ModelUnavailableError(
            f"{handle.provider} call to {handle.model_id} failed: {type(e).__name__}: {e}"
        ) from None
    latency = time.monotonic() - start
    usage: Mapping[str, Any] = result.metrics.accumulated_usage
    tokens_in, tokens_out = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
    return PingResult(
        reply=str(result).strip(),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        latency_s=latency,
        cost_usd=handle.price.cost(tokens_in, tokens_out) if handle.price else None,
    )
