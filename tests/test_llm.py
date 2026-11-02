"""LLM 双通道客户端单测（注入假 OpenAI 客户端，不联网）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wagent.config import LlmSettings
from wagent.llm.client import LLMClient, LLMError, MessageTurn


class FakeCompletions:
    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, results: list) -> None:
        self.completions = FakeCompletions(results)
        self.chat = SimpleNamespace(completions=self.completions)


def make_response(text: str = "在呢", tokens_in: int = 12, tokens_out: int = 4):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def make_client(results: list, settings: LlmSettings | None = None) -> tuple[LLMClient, FakeClient, list[float]]:
    settings = settings or LlmSettings(provider="local", local_model="qwen2.5:7b", max_retries=2)
    fake = FakeClient(results)
    sleeps: list[float] = []
    client = LLMClient(settings, _client=fake, _sleep=sleeps.append)
    return client, fake, sleeps


def messages() -> list[MessageTurn]:
    return [
        MessageTurn("system", "你是本人"),
        MessageTurn("user", "对方刚发来：在吗"),
    ]


def test_local_provider_default() -> None:
    client, _, _ = make_client([make_response()])
    assert client.model == "qwen2.5:7b"
    assert client.settings.provider == "local"


def test_cloud_provider_uses_cloud_model() -> None:
    settings = LlmSettings(provider="cloud", cloud_model="deepseek-chat")
    client, _, _ = make_client([make_response()], settings)
    assert client.model == "deepseek-chat"


def test_chat_returns_text_and_usage() -> None:
    client, _fake, _ = make_client([make_response("八点吧", 20, 6)])
    result = client.chat(messages())
    assert result.text == "八点吧"
    assert result.tokens_in == 20 and result.tokens_out == 6
    assert result.provider == "local"
    assert result.attempts == 1
    assert client.stats.requests == 1
    assert client.stats.total_tokens == 26


def test_payload_uses_settings_defaults() -> None:
    settings = LlmSettings(temperature=0.7, max_tokens=111)
    client, fake, _ = make_client([make_response()], settings)
    client.chat(messages())
    payload = fake.completions.calls[0]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 111
    assert payload["messages"][0] == {"role": "system", "content": "你是本人"}


def test_retry_then_success() -> None:
    client, _, sleeps = make_client([RuntimeError("网络抖动"), make_response()])
    result = client.chat(messages())
    assert result.attempts == 2
    assert sleeps == [1.0]  # 指数退避第一次 2^0
    assert client.stats.failures == 0


def test_retries_exhausted_raises() -> None:
    client, _, sleeps = make_client([RuntimeError("x"), RuntimeError("y"), RuntimeError("z")])
    with pytest.raises(LLMError) as excinfo:
        client.chat(messages())
    assert "3 次" in str(excinfo.value)
    assert client.stats.failures == 1
    assert sleeps == [1.0, 2.0]


def test_missing_usage_defaults_to_zero() -> None:
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    client, _, _ = make_client([resp])
    result = client.chat(messages())
    assert result.tokens_in == 0 and result.tokens_out == 0


def test_empty_content_becomes_empty_string() -> None:
    client, _, _ = make_client(
        [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])]
    )
    assert client.chat(messages()).text == ""
