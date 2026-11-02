"""文本向量化：本地 bge-m3（通过 Ollama 的 OpenAI 兼容接口）。

- 默认模型 ``bge-m3``（1024 维，中文/多语言，本地跑，不出网）；
- 带进程内缓存，重复文本不重复请求；
- 接口用 Protocol，测试时可注入确定性假 embedder。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """向量化协议。"""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """调用本地 Ollama 的 embeddings 接口。"""

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str = "local",
        *,
        dim: int = 1024,
        batch_size: int = 16,
        client=None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=1)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            for text, item in zip(batch, response.data, strict=False):
                self._cache[text] = list(item.embedding)
        return [self._cache[text] for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def cached(self) -> int:
        return len(self._cache)


class HashingEmbedder:
    """零依赖的确定性假 embedder（测试/离线兜底）。

    用字符 n-gram 哈希到固定维度，语义能力远不如 bge-m3，但能让检索逻辑跑起来。
    """

    def __init__(self, dim: int = 64) -> None:
        import hashlib

        self.dim = dim
        self._hashlib = hashlib

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for n in (1, 2, 3):
                for i in range(max(0, len(text) - n + 1)):
                    gram = text[i : i + n]
                    digest = self._hashlib.md5(gram.encode("utf-8")).digest()
                    index = int.from_bytes(digest[:4], "little") % self.dim
                    sign = 1.0 if digest[4] % 2 == 0 else -1.0
                    vec[index] += sign
            norm = sum(value * value for value in vec) ** 0.5 or 1.0
            vectors.append([value / norm for value in vec])
        return vectors
