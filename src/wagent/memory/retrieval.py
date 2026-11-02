"""记忆检索：按「相关性 + 时效 + 重要性」选记忆，装进固定的字符预算里。

对应层级记忆的注入方式：

    【自我画像】        ≤400 字   （L3，常驻）
    【当前联系人画像】  ≤400 字   （L2，常驻）
    【最近 7 天】       ≤400 字   （L1，摘要）
    【相关事实】        ≤600 字   （L4，检索 top-k，带证据）
    【我的说话样本】    ≤300 字   （L5，动态 few-shot）

为什么不用「把聊天记录切块向量化、top-k 塞 prompt」：那样检索到的是**语义相近的
随机片段**，既没有时效、也无法区分「关于他的事实」和「我自己的画像」。这里先用
结构化事实 + 时间衰减，向量检索作为 L5/事件检索的加法。
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wagent.memory.store import MemoryStore, parse_ts
from wagent.memory.types import SELF_PERSON, FactCard

_TERM_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def terms_of(text: str) -> set[str]:
    """粗粒度词项：中文取相邻二元组，英文/数字取整词。"""
    out: set[str] = set()
    for chunk in _TERM_RE.findall(text):
        if chunk.isascii():
            if len(chunk) >= 2:
                out.add(chunk.lower())
        else:
            out.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return out


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """一次记忆检索请求。"""

    person: str
    latest_text: str = ""
    now: datetime = field(default_factory=lambda: datetime.now().astimezone())
    char_budget: int = 1600
    recent_days: int = 7
    max_facts: int = 8
    max_episodes: int = 6
    max_style_samples: int = 6


@dataclass(slots=True)
class MemoryContext:
    """检索结果（可直接拼进 prompt）。"""

    profile_block: str = ""
    facts_block: str = ""
    episodes_block: str = ""
    style_samples: list[str] = field(default_factory=list)
    used_chars: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.profile_block or self.facts_block or self.episodes_block or self.style_samples)

    def to_prompt_block(self) -> str:
        """渲染成 system prompt 里的记忆段落。"""
        sections: list[str] = []
        if self.profile_block:
            sections.append(f"【人物画像】\n{self.profile_block}")
        if self.facts_block:
            sections.append(f"【我记得的事】\n{self.facts_block}")
        if self.episodes_block:
            sections.append(f"【最近发生】\n{self.episodes_block}")
        if self.style_samples:
            sections.append("【我平时怎么说话】\n" + "\n".join(self.style_samples))
        return "\n\n".join(sections)


class MemoryRetriever:
    """分层记忆检索器。"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # ------------------------------------------------------------ 主入口

    def build(self, query: MemoryQuery) -> MemoryContext:
        ctx = MemoryContext()
        budget = max(200, query.char_budget)

        # 1) 画像（自我 + 对方），常驻，按预算截断
        profile_lines: list[str] = []
        self_profile = self.store.get_profile(SELF_PERSON)
        if self_profile is not None:
            profile_lines.append(_render_profile(self_profile, prefix="我"))
        person_profile = self.store.get_profile(query.person)
        if person_profile is not None:
            profile_lines.append(_render_profile(person_profile, prefix="对方"))
        ctx.profile_block = _fit("\n".join(line for line in profile_lines if line), int(budget * 0.3))

        # 2) 相关事实（带时效衰减与重要性）
        facts = self.store.active_facts(query.person, limit=200)
        ranked = sorted(facts, key=lambda f: self._score(f, query), reverse=True)
        chosen = ranked[: query.max_facts]
        ctx.facts_block = _fit(
            "\n".join(f"- {_render_fact(fact)}" for fact in chosen), int(budget * 0.35)
        )

        # 3) 最近 N 天事件
        since = (query.now - timedelta(days=query.recent_days)).isoformat(timespec="seconds")
        episodes = self.store.recent_episodes(query.person, since=since, limit=query.max_episodes)
        ctx.episodes_block = _fit(
            "\n".join(f"- {_short_date(ep.ts)} {ep.summary}" for ep in episodes),
            int(budget * 0.2),
        )

        # 4) 风格样本（按最新消息做种子，随场景变化但可复现）
        samples = self.store.style_samples(SELF_PERSON, limit=50)
        if samples:
            rng = random.Random(_seed(query.latest_text))
            picked = rng.sample(samples, k=min(query.max_style_samples, len(samples)))
            ctx.style_samples = [f"- {s.text}" for s in picked]

        ctx.used_chars = (
            len(ctx.profile_block)
            + len(ctx.facts_block)
            + len(ctx.episodes_block)
            + sum(len(s) + 1 for s in ctx.style_samples)
        )
        return ctx

    # ------------------------------------------------------------ 打分

    def _score(self, fact: FactCard, query: MemoryQuery) -> float:
        latest_terms = terms_of(query.latest_text)
        fact_terms = terms_of(" ".join([fact.subject, fact.predicate, fact.value, fact.evidence_text]))
        relevance = 0.0
        if latest_terms:
            relevance = len(latest_terms & fact_terms) / math.sqrt(len(latest_terms) + 1)
        recency = self._recency(fact.valid_from, query.now)
        importance = max(0.0, min(1.0, fact.confidence))
        return 0.55 * relevance + 0.25 * recency + 0.20 * importance

    @staticmethod
    def _recency(ts: str, now: datetime) -> float:
        happened = parse_ts(ts)
        if happened is None:
            return 0.5
        if happened.tzinfo is None:
            happened = happened.astimezone()
        days = max(0.0, (now - happened).total_seconds() / 86400)
        return 1.0 / (1.0 + days / 30.0)


# ---------------------------------------------------------------- 渲染辅助


def _render_profile(profile, *, prefix: str) -> str:
    name = profile.display_name or ("我" if profile.is_self else "对方")
    head = f"{prefix}({name})"
    parts = [head]
    if profile.relation:
        parts.append(f"关系:{profile.relation}")
    if profile.traits:
        parts.append("特点:" + "、".join(profile.traits))
    if profile.preferences:
        parts.append("偏好:" + "、".join(profile.preferences))
    if profile.taboos:
        parts.append("雷区:" + "、".join(profile.taboos))
    if profile.notes:
        parts.append(profile.notes)
    return "；".join(parts)


def _render_fact(fact: FactCard) -> str:
    text = f"{fact.subject} {fact.predicate}：{fact.value}"
    if fact.confidence < 0.7:
        text += "（不确定）"
    return text


def _short_date(ts: str) -> str:
    happened = parse_ts(ts)
    if happened is None:
        return ts[:10]
    return happened.strftime("%m-%d")


def _seed(text: str) -> int:
    return int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")


def _fit(text: str, limit: int) -> str:
    """按字符预算截断到整行，避免把一句话截一半。"""
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    lines: list[str] = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 > limit:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
