"""Unit tests for the LLM wrapper. LiteLLM is stubbed, so no network or API keys."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ragbench import config, llm


def _fake_response(text: str = "hi") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        model="fake/model-1",
    )


@pytest.fixture
def fake_litellm(monkeypatch):
    calls: list[dict] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _fake_response()

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return _fake_response("async hi")

    stub = SimpleNamespace(
        completion=completion,
        acompletion=acompletion,
        completion_cost=lambda completion_response: 0.0005,
    )
    monkeypatch.setitem(sys.modules, "litellm", stub)
    return stub, calls


@pytest.fixture
def cfg(monkeypatch):
    data = {
        "llm": {
            "generator": {"model": "fake/gen", "temperature": 0.0, "max_tokens": 50},
            "judge": {"model": None},
        }
    }
    monkeypatch.setattr(llm, "get_config", lambda: data)
    return data


def test_complete_returns_result(fake_litellm, cfg):
    _, calls = fake_litellm
    result = llm.complete("What is a tensor?", system="Be brief.")

    assert result.text == "hi"
    assert result.input_tokens == 12
    assert result.output_tokens == 3
    assert result.cost_usd == 0.0005
    assert result.latency_s >= 0
    assert calls[0]["model"] == "fake/gen"
    assert calls[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert calls[0]["messages"][1]["content"] == "What is a tensor?"


def test_overrides_and_passthrough(fake_litellm, cfg):
    _, calls = fake_litellm
    llm.complete("q", model="fake/other", temperature=0.7, response_format={"type": "json_object"})
    assert calls[0]["model"] == "fake/other"
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_missing_model_raises(fake_litellm, cfg):
    with pytest.raises(llm.LLMConfigError):
        llm.complete("q", role="judge")


def test_unpriced_model_gives_none_cost(fake_litellm, cfg):
    stub, _ = fake_litellm

    def no_price(completion_response):
        raise ValueError("no price")

    stub.completion_cost = no_price
    assert llm.complete("q").cost_usd is None


async def test_acomplete(fake_litellm, cfg):
    result = await llm.acomplete("q")
    assert result.text == "async hi"


def test_default_config_loads():
    data = config.load_config()
    assert {"llm", "corpus", "retrieval", "paths"} <= data.keys()
    assert {"generator", "judge"} <= data["llm"].keys()
