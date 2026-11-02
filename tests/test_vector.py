"""向量索引与混合检索单测（用确定性假 embedder，不联网）。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from wagent.memory.embeddings import HashingEmbedder
from wagent.memory.retrieval import MemoryQuery, MemoryRetriever
from wagent.memory.store import MemoryStore
from wagent.memory.types import SELF_PERSON, Episode, FactCard, StyleSample
from wagent.memory.vector import VectorIndex, cosine, pack_vector, rrf, unpack_vector

PERSON = "张伟"
NOW = datetime.fromisoformat("2026-11-02T12:00:00+08:00")


class KeywordEmbedder:
    """把关键词映射到固定维度：搬家/猫/工作/吃饭 各一维。"""

    dim = 4
    TOPICS = (("搬家", "滨江", "搬"), ("猫", "宠物"), ("工作", "实习", "字节"), ("吃饭", "请客"))

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0, 0.0]
            for index, keywords in enumerate(self.TOPICS):
                if any(keyword in text for keyword in keywords):
                    vec[index] = 1.0
            if not any(vec):
                vec = [0.25, 0.25, 0.25, 0.25]
            array = np.asarray(vec, dtype=np.float32)
            out.append((array / np.linalg.norm(array)).tolist())
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


def test_pack_unpack_and_cosine() -> None:
    vector = [0.1, 0.2, 0.3]
    restored = unpack_vector(pack_vector(vector))
    assert np.allclose(restored, vector, atol=1e-6)
    assert abs(cosine(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) - 1.0) < 1e-6
    assert cosine(np.array([1.0, 0, 0]), np.array([0.0, 1, 0])) == 0.0
    assert cosine(np.array([0.0, 0, 0]), np.array([1.0, 0, 0])) == 0.0


def test_rrf_orders_by_fused_rank() -> None:
    fused = rrf([["a", "b"], ["b", "c"]])
    assert fused[0] == "b"  # 两路都靠前
    assert set(fused) == {"a", "b", "c"}


def make_store() -> MemoryStore:
    store = MemoryStore(":memory:")
    store.add_episode(
        Episode(person=PERSON, summary="他下个月搬去滨江，我答应帮忙搬家",
                ts="2026-08-20T20:00:00+08:00")
    )
    store.add_episode(
        Episode(person=PERSON, summary="聊了聊他养的橘猫，胖成球",
                ts="2026-10-28T21:00:00+08:00")
    )
    store.add_style_samples([
        StyleSample(text="行吧 先这样"),
        StyleSample(text="问题不大 我看看"),
        StyleSample(text="搬完请我吃饭"),
    ])
    return store


def test_vector_index_sync_and_search() -> None:
    store = make_store()
    index = VectorIndex(store)
    embedder = KeywordEmbedder()
    added = index.sync(embedder)
    assert added["episode"] == 2
    assert added["style"] == 3
    assert index.count("episode") == 2
    assert index.sync(embedder)["episode"] == 0  # 幂等

    hits = index.search_by_text(PERSON, "搬家的事定在哪天", embedder, kinds=("episode",))
    assert hits and "搬家" in hits[0].text
    assert hits[0].score > 0.9

    style_hits = index.search_by_text(SELF_PERSON, "请我吃饭", embedder, kinds=("style",))
    assert style_hits and "吃饭" in style_hits[0].text


def test_vector_search_respects_person_and_kind() -> None:
    store = make_store()
    store.add_episode(Episode(person="李四", summary="搬家", ts="2026-10-01T10:00:00+08:00"))
    index = VectorIndex(store)
    index.sync(KeywordEmbedder())
    hits = index.search_by_text(PERSON, "搬家", KeywordEmbedder(), kinds=("episode",))
    assert all(hit.item_id for hit in hits)
    assert len(hits) == 1  # 李四的事件不会串到张伟


def test_hybrid_retriever_recalls_old_episode_by_semantics() -> None:
    """60 天前的事件不在「最近 7 天」窗口里，但语义相关，应被向量召回。"""
    store = make_store()
    index = VectorIndex(store)
    embedder = KeywordEmbedder()
    index.sync(embedder)
    retriever = MemoryRetriever(store, vector_index=index, embedder=embedder)
    ctx = retriever.build(
        MemoryQuery(person=PERSON, latest_text="你搬家的事定下来了吗", now=NOW)
    )
    assert "搬家" in ctx.episodes_block
    # recency 也会带出「橘猫」，但语义命中的「搬家」应排在更前面
    assert ctx.episodes_block.index("搬家") < ctx.episodes_block.index("橘猫")


def test_lexical_fallback_without_vector() -> None:
    store = make_store()
    retriever = MemoryRetriever(store)
    ctx = retriever.build(MemoryQuery(person=PERSON, latest_text="搬家", now=NOW))
    # 没有向量时只召回最近 7 天的事件（橘猫），旧事件不出现
    assert "橘猫" in ctx.episodes_block
    assert "搬家" not in ctx.episodes_block


def test_style_samples_vector_recall() -> None:
    store = make_store()
    index = VectorIndex(store)
    embedder = KeywordEmbedder()
    index.sync(embedder)
    retriever = MemoryRetriever(store, vector_index=index, embedder=embedder)
    ctx = retriever.build(MemoryQuery(person=PERSON, latest_text="搬完请你吃饭", now=NOW))
    assert any("吃饭" in line for line in ctx.style_samples)


def test_vector_failure_degrades_gracefully() -> None:
    store = make_store()

    class BrokenEmbedder:
        dim = 3

        def embed(self, texts):
            raise RuntimeError("embedding 服务挂了")

        def embed_one(self, text):
            raise RuntimeError("embedding 服务挂了")

    retriever = MemoryRetriever(store, vector_index=VectorIndex(store), embedder=BrokenEmbedder())
    ctx = retriever.build(MemoryQuery(person=PERSON, latest_text="搬家", now=NOW))
    assert "橘猫" in ctx.episodes_block  # 退回 recency


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=32)
    a = embedder.embed(["搬家的事"])[0]
    b = embedder.embed(["搬家的事"])[0]
    assert a == b
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6
    assert len(a) == 32


def test_embeddings_store_roundtrip() -> None:
    store = MemoryStore(":memory:")
    store.upsert_embedding(
        item_id="ep_1", kind="episode", person=PERSON, text="x", vector=pack_vector([1.0, 0.0]), dim=2
    )
    assert store.count_embeddings("episode") == 1
    rows = store.fetch_embeddings(person=PERSON, kinds=["episode"])
    assert len(rows) == 1 and rows[0]["text"] == "x"
    assert store.embedded_ids("episode") == {"ep_1"}


def test_unembedded_items_for_facts() -> None:
    store = MemoryStore(":memory:")
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="工作", value="字节实习"))
    items = store.unembedded_items("fact")
    assert len(items) == 1
    assert "字节实习" in items[0][2]
    with pytest.raises(ValueError):
        store.unembedded_items("unknown")
