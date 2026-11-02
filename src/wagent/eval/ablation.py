"""消融实验：验证「层级记忆」到底有没有用。

做法：同一批评测对话、同一个本地模型，唯一变量是 **prompt 里有没有记忆块**：
- A 组（baseline）：只有 system 人设 + 风格规则 + few-shot；
- B 组（memory）：额外注入 L0-L5 检索结果（画像/事实/事件/风格样本）。

指标：规则指标（风格相似度/AI 味/长度）+ LLM 盲评胜率 + 记忆命中率。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from wagent.memory.store import MemoryStore
from wagent.memory.types import (
    SELF_PERSON,
    Episode,
    FactCard,
    PersonProfile,
    StyleSample,
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """一条评测样例（来自训练集切分出的 val.jsonl）。"""

    user_content: str
    latest: str = ""
    reference: str = ""


def _extract_latest(user_content: str) -> str:
    for marker in ("对方刚发来：", "对方刚发来:"):
        if marker in user_content:
            return user_content.split(marker, 1)[1].split("\n", 1)[0].strip()
    return ""


def load_cases(path, limit: int = 20) -> list[EvalCase]:
    """从 ChatML JSONL 读取评测样例。"""
    from pathlib import Path

    cases: list[EvalCase] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        messages = row.get("messages", [])
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        if not user or not assistant:
            continue
        cases.append(EvalCase(user, _extract_latest(user), assistant))
        if len(cases) >= limit:
            break
    return cases


def seed_memory_store(store: MemoryStore, *, style_samples: Sequence[str] = ()) -> dict:
    """写入一套用于消融实验的演示记忆（固定内容，保证可复现）。"""
    store.upsert_profile(
        PersonProfile(person=SELF_PERSON, display_name="我", traits=("话少", "偶尔冷幽默"))
    )
    store.upsert_profile(
        PersonProfile(
            person="张伟",
            display_name="张伟",
            relation="大学室友",
            traits=("话痨", "爱打球"),
            taboos=("别提前女友",),
            notes="最近在字节实习，天天加班",
        )
    )
    store.add_fact(
        FactCard(person="张伟", subject="张伟", predicate="计划搬家", value="11月15号搬去滨江",
                 confidence=0.95, evidence_text="搬家定在11月15号 到时候喊你")
    )
    store.add_fact(
        FactCard(person="张伟", subject="张伟", predicate="工作", value="在字节实习，经常加班到十点",
                 confidence=0.9, evidence_text="我最近在字节实习 天天加班到十点")
    )
    store.add_fact(
        FactCard(person="张伟", subject="张伟", predicate="宠物", value="养了两只猫，一只橘一只狸花",
                 confidence=0.9, evidence_text="我养了两只猫 一只橘的一只狸花")
    )
    store.add_episode(
        Episode(person="张伟", summary="约了周日打球，周六他搬家我去帮忙",
                ts="2026-11-01T20:00:00+08:00")
    )
    store.add_episode(
        Episode(person="张伟", summary="他说想换工作，投了字节的实习，怕过不了",
                ts="2026-08-01T20:00:00+08:00")
    )
    store.add_style_samples([StyleSample(text=text) for text in style_samples])
    return store.stats()


def memory_block_for(
    store: MemoryStore,
    person: str,
    latest: str,
    *,
    vector_index=None,
    embedder=None,
) -> str:
    """构造记忆注入块（B 组 prompt 使用；可选向量召回；按意图策略决定是否注入）。"""
    from datetime import datetime

    from wagent.agent.intent import needs_memory
    from wagent.memory.retrieval import MemoryQuery, MemoryRetriever

    if not needs_memory(latest):
        return ""
    retriever = MemoryRetriever(store, vector_index=vector_index, embedder=embedder)
    query = MemoryQuery(person=person, latest_text=latest, now=datetime.now().astimezone())
    return retriever.build(query).to_prompt_block()


def has_memory(content: str) -> bool:
    """记忆命中：prompt 里出现了任意记忆段落。"""
    return any(section in content for section in ("【人物画像】", "【我记得的事】", "【最近发生】", "【我平时怎么说话】"))


def clean_line(text: str) -> str:
    """单行化，方便写报告。"""
    return re.sub(r"\s+", " ", text).strip()
