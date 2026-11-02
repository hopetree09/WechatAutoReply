"""记忆巩固单测（事实抽取 + 冲突/去重 + 摘要）。"""

from __future__ import annotations

from wagent.llm.client import ChatResult
from wagent.memory.consolidate import (
    Consolidator,
    FactExtractor,
    parse_json_array,
)
from wagent.memory.store import MemoryStore
from wagent.memory.types import SELF_PERSON, FactCard, MessageRecord

PERSON = "张三"


class FakeLLM:
    """按 system prompt 决定返回事实 JSON 还是摘要。"""

    def __init__(self, facts_payload: str = "[]", summary: str = "聊了他搬家的事") -> None:
        self.facts_payload = facts_payload
        self.summary = summary
        self.calls: list[list] = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        system = messages[0].content
        text = self.facts_payload if "记忆抽取器" in system else self.summary
        return ChatResult(text, "fake", "fake", 10, 10, 0.01, 1)


def make_messages() -> list[MessageRecord]:
    return [
        MessageRecord(id="msg_a", person=PERSON, speaker="other", text="我下个月搬去滨江",
                      ts="2026-10-28T20:00:00+08:00"),
        MessageRecord(id="msg_b", person=PERSON, speaker="me", text="行 到时候帮你搬",
                      ts="2026-10-28T20:01:00+08:00"),
    ]


def facts_json() -> str:
    return """[
      {"subject": "张伟", "predicate": "计划搬家", "value": "下个月搬滨江",
       "evidence_id": "msg_a", "confidence": 0.9, "about": "other"},
      {"subject": "我", "predicate": "答应帮忙搬家", "value": "到时候帮张伟搬",
       "evidence_id": "msg_b", "confidence": 0.85, "about": "me"}
    ]"""


def test_parse_json_array_tolerates_fences_and_noise() -> None:
    assert parse_json_array('```json\n[{"a":1}]\n```') == [{"a": 1}]
    assert parse_json_array('好的，结果如下：[{"a":1}] 完毕') == [{"a": 1}]
    assert parse_json_array("不是 JSON") == []
    assert parse_json_array("") == []


def test_extract_creates_cards_with_evidence() -> None:
    extractor = FactExtractor(FakeLLM(facts_json()))
    cards = extractor.extract(PERSON, make_messages())
    assert len(cards) == 2
    other_card = next(c for c in cards if c.person == PERSON)
    self_card = next(c for c in cards if c.person == SELF_PERSON)
    assert other_card.predicate == "计划搬家"
    assert other_card.evidence_msg_ids == ("msg_a",)
    assert "滨江" in other_card.evidence_text
    assert self_card.subject == "我"


def test_extract_drops_cards_without_valid_evidence() -> None:
    payload = """[
      {"subject": "张伟", "predicate": "喜欢猫", "value": "养了两只猫",
       "evidence_id": "msg_不存在", "confidence": 0.9, "about": "other"}
    ]"""
    cards = FactExtractor(FakeLLM(payload)).extract(PERSON, make_messages())
    assert cards == []


def test_extract_handles_malformed_output() -> None:
    assert FactExtractor(FakeLLM("抱歉我不能")).extract(PERSON, make_messages()) == []


def test_extract_requires_core_fields() -> None:
    payload = '[{"subject": "", "predicate": "x", "value": "y", "evidence_id": "msg_a"}]'
    assert FactExtractor(FakeLLM(payload)).extract(PERSON, make_messages()) == []


def test_extract_caps_max_facts() -> None:
    items = ",".join(
        f'{{"subject":"张伟","predicate":"事实{i}","value":"v{i}","evidence_id":"msg_a","confidence":0.9,"about":"other"}}'
        for i in range(20)
    )
    cards = FactExtractor(FakeLLM(f"[{items}]"), max_facts=3).extract(PERSON, make_messages())
    assert len(cards) == 3


# ---------------------------------------------------------------- 巩固流程


def test_consolidate_writes_facts_and_episode() -> None:
    store = MemoryStore(":memory:")
    llm = FakeLLM(facts_json(), summary="他下个月搬家，我答应帮忙")
    consolidator = Consolidator(store, FactExtractor(llm))
    result = consolidator.consolidate_person(PERSON, messages=make_messages())

    assert result.facts_added == 2
    assert result.episode_id
    assert result.summary.startswith("他下个月搬家")
    assert len(store.active_facts(PERSON)) == 1
    assert len(store.active_facts(SELF_PERSON)) == 1
    assert len(store.recent_episodes(PERSON)) == 1
    assert store.get_meta("last_consolidate_at")


def test_consolidate_skips_same_value_and_supersedes_changed_value() -> None:
    store = MemoryStore(":memory:")
    store.add_fact(
        FactCard(person=PERSON, subject="张伟", predicate="计划搬家", value="下个月搬滨江",
                 confidence=0.9, valid_from="2026-10-01T00:00:00+08:00")
    )
    llm = FakeLLM(facts_json())
    result = Consolidator(store, FactExtractor(llm), summarize=False).consolidate_person(
        PERSON, messages=make_messages()
    )
    # 「计划搬家」同值 → 跳过；「我答应帮忙搬家」→ 新增
    assert result.facts_skipped == 1
    assert result.facts_added == 1
    assert len(store.active_facts(PERSON)) == 1

    changed = """[{"subject":"张伟","predicate":"计划搬家","value":"已经搬完，滨江",
                  "evidence_id":"msg_a","confidence":0.9,"about":"other"}]"""
    result2 = Consolidator(store, FactExtractor(FakeLLM(changed)), summarize=False).consolidate_person(
        PERSON, messages=make_messages()
    )
    assert result2.facts_added == 1
    active = store.active_facts(PERSON)
    assert [f.value for f in active] == ["已经搬完，滨江"]


def test_consolidate_without_messages_is_noop() -> None:
    store = MemoryStore(":memory:")
    result = Consolidator(store, FactExtractor(FakeLLM())).consolidate_person(PERSON)
    assert result.facts_added == 0
    assert result.episode_id == ""


def test_summary_failure_does_not_break_fact_writing() -> None:
    class SummaryBroken(FakeLLM):
        def chat(self, messages, **kwargs):
            if "聊天摘要器" in messages[0].content:
                raise RuntimeError("摘要挂了")
            return super().chat(messages, **kwargs)

    store = MemoryStore(":memory:")
    result = Consolidator(store, FactExtractor(SummaryBroken(facts_json()))).consolidate_person(
        PERSON, messages=make_messages()
    )
    assert result.facts_added == 2
    assert result.summary == ""
