"""向量索引 + 倒数排名融合（RRF）。

为什么不是「纯向量检索」：
- 最近发生的事（recency）和语义相关的事（vector）各有价值，单一排序都会漏；
- 用 RRF 融合两路排名，不需要调权重，工程上更稳。

存储：SQLite 里存 float32 BLOB（1024 维 = 4KB/条），数据量在个人聊天规模（万级）下
直接内存暴力算余弦即可，不需要额外向量库。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from wagent.memory.embeddings import Embedder
from wagent.memory.store import MemoryStore

KINDS = ("episode", "style", "fact")


def pack_vector(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


def rrf(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    """倒数排名融合：``score = Σ 1/(k + rank)``，返回按分数降序的 id。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]


@dataclass(frozen=True, slots=True)
class VectorHit:
    item_id: str
    kind: str
    text: str
    score: float


class VectorIndex:
    """记忆库的向量索引（按 person + kind 过滤）。"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def count(self, kind: str | None = None) -> int:
        return self.store.count_embeddings(kind)

    def sync(
        self, embedder: Embedder, *, kinds: tuple[str, ...] = KINDS, limit: int = 500
    ) -> dict[str, int]:
        """把还没索引的 episode/style/fact 批量向量化。返回各 kind 新增数量。"""
        added: dict[str, int] = {}
        for kind in kinds:
            items = self.store.unembedded_items(kind, limit=limit)
            if not items:
                added[kind] = 0
                continue
            vectors = embedder.embed([text for _, _, text in items])
            for (item_id, person, text), vector in zip(items, vectors, strict=False):
                self.store.upsert_embedding(
                    item_id=item_id,
                    kind=kind,
                    person=person,
                    text=text,
                    vector=pack_vector(vector),
                    dim=len(vector),
                )
            added[kind] = len(items)
        return added

    def search(
        self,
        person: str,
        query_vector: list[float] | np.ndarray,
        *,
        kinds: tuple[str, ...] = ("episode", "style"),
        top_k: int = 10,
        min_score: float = 0.12,
    ) -> list[VectorHit]:
        rows = self.store.fetch_embeddings(person=person, kinds=list(kinds))
        if not rows:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        hits: list[VectorHit] = []
        for row in rows:
            score = cosine(query, unpack_vector(row["vector"]))
            if score < min_score:
                continue
            hits.append(VectorHit(row["item_id"], row["kind"], row["text"], score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def search_by_text(
        self,
        person: str,
        text: str,
        embedder: Embedder,
        *,
        kinds: tuple[str, ...] = ("episode", "style"),
        top_k: int = 10,
    ) -> list[VectorHit]:
        if not text.strip():
            return []
        return self.search(person, embedder.embed_one(text), kinds=kinds, top_k=top_k)


Kind = Literal["episode", "style", "fact"]
