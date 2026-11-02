"""记忆层数据结构。

设计要点（对应「精细人物画像」）：
- 画像不是一段散文，而是**结构化字段 + 原子事实卡**：
  事实卡带 ``subject/predicate/value``、有效期、置信度、**证据消息 id 与原文**，
  这样才能更新、冲突消解、溯源、按需删除；
- 事件（episode）按时间存摘要，解决「12 条消息记不住上周的事」；
- 风格样本单独存，用来做动态 few-shot。
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime

SELF_PERSON = "__self__"

#: 自我画像在 profiles 表里的主键
SELF_KEY = SELF_PERSON


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def text_fingerprint(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """一条落库的历史消息（离线导入 + 在线增量都用它）。"""

    person: str
    speaker: str  # me | other
    text: str
    ts: str
    session: str = ""
    source: str = "import"
    id: str = field(default_factory=lambda: new_id("msg_"))

    @property
    def fingerprint(self) -> str:
        return text_fingerprint(self.person, self.speaker, self.text, self.ts)


@dataclass(frozen=True, slots=True)
class FactCard:
    """原子事实卡：可更新、可溯源、有时效。"""

    person: str
    subject: str
    predicate: str
    value: str
    confidence: float = 0.8
    valid_from: str = field(default_factory=now_iso)
    valid_to: str | None = None
    status: str = "active"  # active | superseded | deleted
    evidence_msg_ids: tuple[str, ...] = ()
    evidence_text: str = ""
    source: str = "own_message"  # own_message | other_message | manual
    id: str = field(default_factory=lambda: new_id("fact_"))

    def key(self) -> tuple[str, str, str]:
        """同一 (person, subject, predicate) 只应有一条 active 事实。"""
        return (self.person, self.subject, self.predicate)


@dataclass(frozen=True, slots=True)
class Episode:
    """短期记忆：某个时间点发生了什么（用于「最近一周」）。"""

    person: str
    summary: str
    ts: str
    topics: tuple[str, ...] = ()
    msg_ids: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: new_id("ep_"))


@dataclass(frozen=True, slots=True)
class PersonProfile:
    """人物画像（person="__self__" 表示自我画像）。"""

    person: str
    display_name: str = ""
    relation: str = ""
    traits: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    taboos: tuple[str, ...] = ()
    notes: str = ""
    updated_at: str = field(default_factory=now_iso)

    @property
    def is_self(self) -> bool:
        return self.person == SELF_PERSON


@dataclass(frozen=True, slots=True)
class StyleSample:
    """我真实说过的一句话（动态 few-shot 来源）。"""

    text: str
    person: str = SELF_PERSON
    scene: str = ""
    ts: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("sty_"))
