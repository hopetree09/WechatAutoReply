"""记忆巩固：把聊天消息抽取成结构化事实卡 + 生成短期事件摘要。

这是「层级记忆」从"存了消息"到"记得住事"的关键一步：

    原始消息 ──LLM 抽取──▶ 原子事实卡（subject/predicate/value/证据id/置信度）
                              │
                              ├─ 去重：同 (subject,predicate) 同值 → 跳过
                              ├─ 冲突：同 (subject,predicate) 不同值 → 旧事实 supersede
                              └─ 防幻觉：没有 evidence_msg_id / 证据不在输入里 → 丢弃

设计约束：
1. **只从证据里抽**：返回的每条事实必须指向输入里真实存在的消息 id，否则丢弃；
2. **区分「关于我」和「关于对方」**：关于我 → 存到 ``__self__`` 的画像；
3. LLM 输出必须是 JSON；解析失败/格式错误一律降级为空结果，不中断主流程。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from wagent.llm.client import ChatModel, MessageTurn
from wagent.memory.store import MemoryStore
from wagent.memory.types import (
    SELF_PERSON,
    Episode,
    FactCard,
    MessageRecord,
    now_iso,
)

EXTRACT_SYSTEM = """你是记忆抽取器。从聊天记录中抽取**可以长期记住**的事实，输出 JSON。

规则：
- 只抽稳定、可复用的事实：身份、关系、工作/学习、偏好、计划、约定、重要经历、雷区。
- 不要抽临时寒暄（"在吗""哈哈"）、不要推测、不要编造。
- 每条事实必须引用它来自哪条消息的 id（evidence_id 只能取输入里出现的 id）。
- about 字段：这条事实描述的是「我」(me) 还是「对方」(other)。
- 置信度 confidence：0~1，原话明确才给 0.8 以上。
- 最多 8 条，没有就返回空数组。

只输出 JSON 数组，不要解释：
[{"subject":"张伟","predicate":"计划搬家","value":"10月底搬到滨江","evidence_id":"msg_xxx","confidence":0.9,"about":"other"}]"""

SUMMARY_SYSTEM = """你是聊天摘要器。把给定聊天记录压缩成 1~2 句中文摘要，只保留事件、情绪、约定、待办。
只输出摘要本身，不要任何前缀。"""


@dataclass(frozen=True, slots=True)
class FactCandidate:
    subject: str
    predicate: str
    value: str
    evidence_id: str
    about: str = "other"  # me | other
    confidence: float = 0.8


@dataclass
class ConsolidateResult:
    person: str = ""
    facts_added: int = 0
    facts_skipped: int = 0
    facts_rejected: int = 0  # 无证据/格式非法
    episode_id: str = ""
    summary: str = ""

    def describe(self) -> str:
        return (
            f"巩固 {self.person}：新增事实 {self.facts_added}，"
            f"跳过重复 {self.facts_skipped}，丢弃无证据 {self.facts_rejected}，"
            f"摘要={'有' if self.summary else '无'}"
        )


def _render_messages(messages: Sequence[MessageRecord]) -> str:
    rows = []
    for message in messages:
        who = "我" if message.speaker == "me" else "对方"
        rows.append(f"[{message.id}] {who}: {message.text}")
    return "\n".join(rows)


def parse_json_array(raw: str) -> list[dict[str, Any]]:
    """从模型输出里抠出 JSON 数组（容忍 ```json 代码块与前后废话）。"""
    text = (raw or "").strip()
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


@dataclass
class FactExtractor:
    """用 LLM 从消息里抽事实；输出做严格校验。"""

    llm: ChatModel
    max_facts: int = 8

    def extract(self, person: str, messages: Sequence[MessageRecord]) -> list[FactCard]:
        if not messages:
            return []
        user = (
            "聊天记录（每行开头的方括号是消息 id）：\n"
            f"{_render_messages(messages)}\n\n"
            "请抽取长期记忆事实，输出 JSON 数组。"
        )
        result = self.llm.chat(
            [MessageTurn("system", EXTRACT_SYSTEM), MessageTurn("user", user)],
            temperature=0.1,
            max_tokens=800,
        )
        return self._to_cards(person, result.text, messages)

    def _to_cards(
        self, person: str, raw: str, messages: Sequence[MessageRecord]
    ) -> list[FactCard]:
        by_id = {m.id: m for m in messages}
        cards: list[FactCard] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for item in parse_json_array(raw):
            subject = str(item.get("subject", "")).strip()[:40]
            predicate = str(item.get("predicate", "")).strip()[:60]
            value = str(item.get("value", "")).strip()[:200]
            evidence_id = str(item.get("evidence_id", "")).strip()
            about = str(item.get("about", "other")).strip().lower()
            if not (subject and predicate and value):
                continue
            evidence = by_id.get(evidence_id)
            if evidence is None:  # 证据不存在 / 模型编造 → 丢弃
                continue
            if about not in {"me", "other"}:
                about = "other"
            confidence = _clamp(float(item.get("confidence", 0.8) or 0.8))
            owner = SELF_PERSON if about == "me" else person
            key = (owner, subject, predicate, value)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cards.append(
                FactCard(
                    person=owner,
                    subject=subject if about == "other" else "我",
                    predicate=predicate,
                    value=value,
                    confidence=confidence,
                    evidence_msg_ids=(evidence.id,),
                    evidence_text=evidence.text[:300],
                    source="own_message" if about == "me" else "other_message",
                )
            )
            if len(cards) >= self.max_facts:
                break
        return cards


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class Consolidator:
    """把消息 → 事实/事件写进记忆库（幂等：同值不重复写）。"""

    store: MemoryStore
    extractor: FactExtractor
    summarize: bool = True
    _now: Any = field(default=None, repr=False)

    def consolidate_person(
        self, person: str, *, limit: int = 200, messages: Sequence[MessageRecord] | None = None
    ) -> ConsolidateResult:
        records = list(messages) if messages is not None else self.store.recent_messages(person, limit=limit)
        result = ConsolidateResult(person=person)
        if not records:
            return result

        cards = self.extractor.extract(person, records)
        active = {
            (fact.subject, fact.predicate): fact
            for fact in self.store.active_facts(person, limit=500)
        }
        self_active = {
            (fact.subject, fact.predicate): fact
            for fact in self.store.active_facts(SELF_PERSON, limit=500)
        }

        for card in cards:
            bucket = self_active if card.person == SELF_PERSON else active
            existing = bucket.get((card.subject, card.predicate))
            if existing is not None and existing.value.strip() == card.value.strip():
                result.facts_skipped += 1
                continue
            if not card.evidence_msg_ids:
                result.facts_rejected += 1
                continue
            self.store.add_fact(card)
            result.facts_added += 1

        if self.summarize:
            summary = self._summarize(records)
            if summary:
                episode = Episode(
                    person=person,
                    summary=summary,
                    ts=now_iso(),
                    topics=(),
                    msg_ids=tuple(record.id for record in records[:50]),
                )
                self.store.add_episode(episode)
                result.episode_id = episode.id
                result.summary = summary

        self.store.set_meta("last_consolidate_at", now_iso())
        return result

    def _summarize(self, records: Sequence[MessageRecord]) -> str:
        text = "\n".join(
            f"{'我' if r.speaker == 'me' else '对方'}: {r.text}" for r in records[-60:]
        )
        try:
            result = self.llm.chat(
                [
                    MessageTurn("system", SUMMARY_SYSTEM),
                    MessageTurn("user", f"聊天记录：\n{text}"),
                ],
                temperature=0.3,
                max_tokens=200,
            )
        except Exception:
            return ""
        summary = result.text.strip()
        return summary[:200]

    @property
    def llm(self) -> ChatModel:
        return self.extractor.llm
