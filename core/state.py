"""会话状态与上下文管理。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    speaker: str      # "me" / "other"
    text: str


@dataclass
class Conversation:
    """一个会话的上下文，内置文本去重。"""

    name: str = ""
    messages: list[Message] = field(default_factory=list)
    _seen: set = field(default_factory=set)   # 已见消息指纹，用于去重

    def update(self, new_msgs: list[Message]) -> bool:
        """增量合并新消息，返回是否有新增。"""
        changed = False
        for m in new_msgs:
            text = m.text.strip()
            if not text:
                continue
            key = (m.speaker, text)
            if key in self._seen:
                continue
            self._seen.add(key)
            self.messages.append(m)
            changed = True
        return changed

    def last_other_text(self) -> Optional[str]:
        """最近一条对方消息文本。"""
        for m in reversed(self.messages):
            if m.speaker == "other":
                return m.text
        return None

    def context_texts(self, limit: int = 12) -> list[str]:
        """最近 N 条消息文本，用于喂给 LLM。"""
        return [m.text for m in self.messages[-limit:]]
