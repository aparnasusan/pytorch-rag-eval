"""Provider-agnostic LLM wrapper built on LiteLLM.

Every model call in the project goes through `complete` / `acomplete`, so switching
between Claude, OpenAI and Ollama is a one-line change in config.yaml, and every call
reports tokens, cost and latency in the same shape.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from ragbench.config import get_config

Role = Literal["generator", "judge"]
Messages = list[dict[str, str]]


class LLMConfigError(ValueError):
    """Raised when a role has no model configured."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when LiteLLM has no price for the model
    latency_s: float


def _to_messages(prompt: str | Messages, system: str | None) -> Messages:
    messages: Messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    )
    if system is not None:
        messages = [{"role": "system", "content": system}, *messages]
    return messages


def _build_kwargs(role: Role, overrides: dict[str, Any]) -> dict[str, Any]:
    settings = dict(get_config().get("llm", {}).get(role, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    if not settings.get("model"):
        raise LLMConfigError(
            f"No model set for llm.{role}.model in config.yaml (or pass model=...)."
        )
    kwargs: dict[str, Any] = {
        "model": settings.pop("model"),
        "temperature": settings.pop("temperature", 0.0),
        "max_tokens": settings.pop("max_tokens", 1024),
        "timeout": settings.pop("timeout_s", 60),
        "num_retries": settings.pop("num_retries", 3),
    }
    kwargs.update(settings)  # anything else (e.g. response_format) passes through
    return kwargs


def _to_result(litellm: Any, response: Any, model: str, latency_s: float) -> LLMResult:
    usage = getattr(response, "usage", None)
    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:  # unpriced/local models
        cost = None
    return LLMResult(
        text=response.choices[0].message.content or "",
        model=getattr(response, "model", None) or model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=cost,
        latency_s=latency_s,
    )


def complete(
    prompt: str | Messages,
    *,
    role: Role = "generator",
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **extra: Any,
) -> LLMResult:
    """Synchronous completion using the model configured for `role`."""
    import litellm  # imported lazily: slow import, and lets tests stub it

    kwargs = _build_kwargs(
        role, {"model": model, "temperature": temperature, "max_tokens": max_tokens, **extra}
    )
    start = time.perf_counter()
    response = litellm.completion(messages=_to_messages(prompt, system), **kwargs)
    return _to_result(litellm, response, kwargs["model"], time.perf_counter() - start)


async def acomplete(
    prompt: str | Messages,
    *,
    role: Role = "generator",
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **extra: Any,
) -> LLMResult:
    """Async completion; the eval runner uses this with capped concurrency."""
    import litellm

    kwargs = _build_kwargs(
        role, {"model": model, "temperature": temperature, "max_tokens": max_tokens, **extra}
    )
    start = time.perf_counter()
    response = await litellm.acompletion(messages=_to_messages(prompt, system), **kwargs)
    return _to_result(litellm, response, kwargs["model"], time.perf_counter() - start)
