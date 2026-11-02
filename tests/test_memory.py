"""记忆层单测：SQLite 存储 + 分层检索。"""

from __future__ import annotations

from datetime import datetime

from wagent.memory.retrieval import MemoryQuery, MemoryRetriever, terms_of
from wagent.memory.store import MemoryStore
from wagent.memory.types import (
    SELF_PERSON,
    Episode,
    FactCard,
    MessageRecord,
    PersonProfile,
    StyleSample,
)

PERSON = "张三"


def make_store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------- 存储


def test_add_messages_dedupes_by_fingerprint() -> None:
    store = make_store()
    msg = MessageRecord(person=PERSON, speaker="other", text="在吗", ts="2026-11-01T10:00:00+08:00")
    assert store.add_messages([msg]) == 1
    assert store.add_messages([msg]) == 0
    assert store.stats()["messages"] == 1


def test_recent_messages_ordered() -> None:
    store = make_store()
    store.add_messages(
        [
            MessageRecord(person=PERSON, speaker="other", text="第一条", ts="2026-11-01T10:00:00+08:00"),
            MessageRecord(person=PERSON, speaker="me", text="第二条", ts="2026-11-01T10:01:00+08:00"),
            MessageRecord(person="李四", speaker="other", text="别人的消息", ts="2026-11-01T11:00:00+08:00"),
        ]
    )
    recent = store.recent_messages(PERSON, limit=5)
    assert [m.text for m in recent] == ["第一条", "第二条"]


def test_fact_supersede_on_same_key() -> None:
    """同一 (person, subject, predicate) 新值应把旧值标记 supersede，并记录有效期。"""
    store = make_store()
    old = FactCard(person=PERSON, subject="张伟", predicate="计划搬家", value="10月底搬滨江",
                   valid_from="2026-09-28T10:00:00+08:00")
    store.add_fact(old)
    new = FactCard(person=PERSON, subject="张伟", predicate="计划搬家", value="已搬完，滨江",
                   valid_from="2026-11-01T10:00:00+08:00")
    store.add_fact(new)

    active = store.active_facts(PERSON)
    assert len(active) == 1
    assert active[0].value == "已搬完，滨江"

    row = store.conn.execute("SELECT status, valid_to FROM facts WHERE id=?", (old.id,)).fetchone()
    assert row["status"] == "superseded"
    assert row["valid_to"]


def test_different_predicates_both_active() -> None:
    store = make_store()
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="工作", value="后端"))
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="爱好", value="羽毛球"))
    assert len(store.active_facts(PERSON)) == 2


def test_search_facts_by_keyword() -> None:
    store = make_store()
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="计划搬家", value="月底搬滨江",
                            evidence_text="我下个月搬去滨江"))
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="工作", value="字节跳动"))
    hits = store.search_facts(PERSON, ["搬家"])
    assert [f.predicate for f in hits] == ["计划搬家"]

    store2 = make_store()
    store2.add_fact(FactCard(person=PERSON, subject="张伟", predicate="工作", value="字节跳动"))
    assert store2.search_facts(PERSON, ["滨江"]) == []


def test_delete_fact() -> None:
    store = make_store()
    fact = FactCard(person=PERSON, subject="张伟", predicate="工作", value="字节")
    store.add_fact(fact)
    assert store.delete_fact(fact.id) is True
    assert store.active_facts(PERSON) == []


def test_profile_roundtrip() -> None:
    store = make_store()
    store.upsert_profile(
        PersonProfile(
            person=PERSON,
            display_name="张伟",
            relation="大学室友",
            traits=("话痨", "爱抬杠"),
            preferences=("羽毛球",),
            taboos=("别提早恋",),
            notes="最近在找工作",
        )
    )
    profile = store.get_profile(PERSON)
    assert profile is not None
    assert profile.display_name == "张伟"
    assert profile.taboos == ("别提早恋",)

    # 覆盖更新
    store.upsert_profile(PersonProfile(person=PERSON, display_name="张伟", notes="已入职"))
    assert store.get_profile(PERSON).notes == "已入职"  # type: ignore[union-attr]


def test_episodes_and_style_samples() -> None:
    store = make_store()
    store.add_episode(Episode(person=PERSON, summary="聊了他搬家的事", ts="2026-11-01T20:00:00+08:00",
                              topics=("搬家",)))
    assert len(store.recent_episodes(PERSON)) == 1

    samples = [StyleSample(text="行吧 先这样"), StyleSample(text="问题不大")]
    assert store.add_style_samples(samples) == 2
    assert store.add_style_samples(samples) == 0  # 去重
    assert len(store.style_samples()) == 2


# ---------------------------------------------------------------- 检索


def _seed_store() -> MemoryStore:
    store = make_store()
    store.upsert_profile(
        PersonProfile(person=SELF_PERSON, display_name="我", traits=("话少",), notes="后端工程师")
    )
    store.upsert_profile(
        PersonProfile(person=PERSON, display_name="张伟", relation="室友",
                      traits=("话痨",), taboos=("别提早恋",))
    )
    store.add_fact(
        FactCard(person=PERSON, subject="张伟", predicate="计划搬家", value="月底搬滨江",
                 confidence=0.9, valid_from="2026-10-30T10:00:00+08:00",
                 evidence_text="我下个月搬去滨江")
    )
    store.add_fact(
        FactCard(person=PERSON, subject="张伟", predicate="工作", value="在字节实习",
                 confidence=0.9, valid_from="2026-08-01T10:00:00+08:00")
    )
    store.add_episode(
        Episode(person=PERSON, summary="他吐槽了实习加班", ts="2026-11-01T21:00:00+08:00")
    )
    store.add_style_samples([StyleSample(text="行吧 先这样"), StyleSample(text="问题不大")])
    return store


def test_build_returns_all_layers() -> None:
    store = _seed_store()
    ctx = MemoryRetriever(store).build(
        MemoryQuery(person=PERSON, latest_text="你搬家搬完了吗", now=datetime.fromisoformat("2026-11-02T12:00:00+08:00"))
    )
    assert "我(我)" in ctx.profile_block or "后端工程师" in ctx.profile_block
    assert "张伟" in ctx.profile_block
    assert "搬家" in ctx.facts_block
    assert "吐槽" in ctx.episodes_block
    assert ctx.style_samples
    assert ctx.used_chars > 0
    block = ctx.to_prompt_block()
    assert "【人物画像】" in block and "【我记得的事】" in block


def test_relevance_ranks_matching_fact_first() -> None:
    store = _seed_store()
    retriever = MemoryRetriever(store)
    query = MemoryQuery(
        person=PERSON,
        latest_text="搬家的事情定了吗",
        now=datetime.fromisoformat("2026-11-02T12:00:00+08:00"),
        max_facts=1,
    )
    ctx = retriever.build(query)
    assert "搬家" in ctx.facts_block


def test_recency_affects_score() -> None:
    store = make_store()
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="旧事", value="很久以前的事",
                            confidence=0.9, valid_from="2025-01-01T00:00:00+08:00"))
    store.add_fact(FactCard(person=PERSON, subject="张伟", predicate="新事", value="最近的事",
                            confidence=0.9, valid_from="2026-11-01T00:00:00+08:00"))
    retriever = MemoryRetriever(store)
    query = MemoryQuery(person=PERSON, latest_text="随便聊聊",
                        now=datetime.fromisoformat("2026-11-02T12:00:00+08:00"), max_facts=1)
    ctx = retriever.build(query)
    assert "最近的事" in ctx.facts_block


def test_budget_is_respected() -> None:
    store = make_store()
    for i in range(50):
        store.add_fact(
            FactCard(person=PERSON, subject="张伟", predicate=f"事实{i}",
                     value="很长的值" * 20, confidence=0.8)
        )
    ctx = MemoryRetriever(store).build(
        MemoryQuery(person=PERSON, latest_text="随便", char_budget=300, max_facts=50)
    )
    assert len(ctx.facts_block) <= 300
    assert ctx.used_chars <= 1200  # 各分块上限之和


def test_no_memory_gives_empty_context() -> None:
    ctx = MemoryRetriever(make_store()).build(MemoryQuery(person="陌生人"))
    assert ctx.is_empty
    assert ctx.to_prompt_block() == ""


def test_terms_of_extracts_cjk_bigrams() -> None:
    terms = terms_of("你搬家了吗 AB test")
    assert "搬家" in terms
    assert "ab" in terms
    assert "test" in terms


def test_style_sample_sampling_is_deterministic() -> None:
    store = make_store()
    store.add_style_samples([StyleSample(text=f"样本{i}") for i in range(20)])
    retriever = MemoryRetriever(store)
    now = datetime.now().astimezone()
    a = retriever.build(MemoryQuery(person=PERSON, latest_text="同一句话", now=now))
    b = retriever.build(MemoryQuery(person=PERSON, latest_text="同一句话", now=now))
    assert a.style_samples == b.style_samples
