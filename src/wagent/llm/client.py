"""模型层：本地（默认）/ 云端双通道 OpenAI 兼容客户端。

设计目标：
- 一套接口切换本地 Ollama / 云端 DeepSeek（隐私优先，默认本地）；
- 超时 + 指数退避重试，避免网络抖动导致静默漏消息；
- 统一记录 token / 耗时 / 估算成本，给可观测性打底；
- 客户端可注入，便于单测（不依赖网络）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from wagent.config import LlmSettings


class LLMError(RuntimeError):
    """重试耗尽后仍然失败。"""


@dataclass(slots=True)
class UsageStats:
    """累计用量与成本统计。"""

    requests: int = 0
    failures: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def record(self, tokens_in: int, tokens_out: int) -> None:
        self.requests += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    def record_failure(self) -> None:
        self.failures += 1

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    attempts: int


@dataclass(frozen=True, slots=True)
class MessageTurn:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMClient:
    """OpenAI 兼容 Chat Completions 客户端（本地/云端）。"""

    settings: LlmSettings
    _client: Any = None
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    stats: UsageStats = field(default_factory=UsageStats, repr=False)

    def __post_init__(self) -> None:
        if self._client is None:
            from openai import OpenAI

            base_url, api_key = self._credentials()
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.settings.timeout_s,
                max_retries=0,  # 重试逻辑自己控制，便于统计
            )

    def _credentials(self) -> tuple[str, str]:
        if self.settings.provider == "local":
            return self.settings.local_base_url, self.settings.local_api_key
        return self.settings.cloud_base_url, self.settings.cloud_api_key

    @property
    def model(self) -> str:
        return (
            self.settings.local_model
            if self.settings.provider == "local"
            else self.settings.cloud_model
        )

    def chat(
        self,
        messages: Sequence[MessageTurn],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """调用模型；失败按指数退避重试，最终失败抛 :class:`LLMError`。"""
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }

        started = self._clock()
        last_error: Exception | None = None
        attempts = self.settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.chat.completions.create(**payload)
                text = (resp.choices[0].message.content or "").strip()
                tokens_in, tokens_out = self._tokens_of(resp)
                self.stats.record(tokens_in, tokens_out)
                return ChatResult(
                    text=text,
                    model=self.model,
                    provider=self.settings.provider,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_s=round(self._clock() - started, 3),
                    attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    self._sleep(min(2.0 ** (attempt - 1), 8.0))

        self.stats.record_failure()
        raise LLMError(f"{self.settings.provider}/{self.model} 调用失败（{attempts} 次）: {last_error}")

    @staticmethod
    def _tokens_of(resp: Any) -> tuple[int, int]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def close(self) -> None:
        """尽量释放底层连接（OpenAI SDK 有 close 时调用）。"""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
