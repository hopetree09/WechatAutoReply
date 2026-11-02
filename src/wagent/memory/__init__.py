"""记忆层：L0 工作记忆 → L5 风格样本库。

- L0 工作记忆：当前会话最近若干条消息（放在上下文里，不落库）
- L1 短期记忆：最近 7 天的事件摘要（``episodes``）
- L2 人物画像：每个联系人一份结构化画像（``profiles``）
- L3 自我画像：``profiles`` 里 person="__self__" 的那条
- L4 事实库：可更新、可溯源、有时效的原子事实卡（``facts``）
- L5 风格样本：真实说过的句子（``style_samples``，后续可加向量检索）
"""

from wagent.memory.consolidate import Consolidator, FactCandidate, FactExtractor
from wagent.memory.retrieval import MemoryContext, MemoryQuery, MemoryRetriever
from wagent.memory.store import MemoryStore
from wagent.memory.types import Episode, FactCard, MessageRecord, PersonProfile, StyleSample

__all__ = [
    "Consolidator",
    "Episode",
    "FactCandidate",
    "FactCard",
    "FactExtractor",
    "MemoryContext",
    "MemoryQuery",
    "MemoryRetriever",
    "MemoryStore",
    "MessageRecord",
    "PersonProfile",
    "StyleSample",
]
