"""IM 适配层：把「屏幕」抽象成可替换的聊天通道。

- :class:`Frame` 是一帧感知结果（消息 + 会话名 + 可选截图）；
- :class:`IMAdapter` 协议让主循环不关心是「真实微信窗口」还是「模拟器」，
  这样评测/CI 可以不碰真实账号，也为后面接别的 IM 留了口子。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from wagent.perception.locator import InputBox
from wagent.perception.types import Message


@dataclass(slots=True)
class Frame:
    """一帧感知结果。"""

    messages: list[Message] = field(default_factory=list)
    session: str = ""
    image: np.ndarray | None = field(default=None, repr=False)

    @property
    def bottom(self) -> Message | None:
        return self.messages[-1] if self.messages else None


@runtime_checkable
class IMAdapter(Protocol):
    """聊天通道协议。"""

    @property
    def hwnd(self) -> int:
        """窗口句柄；模拟器可返回 0。"""
        ...

    def capture(self) -> Frame:
        """截一帧并解析成结构化消息。"""
        ...

    def input_box(self) -> InputBox | None:
        """最近一次定位到的输入框（可能为 None，表示还没截过图）。"""
        ...

    def close(self) -> None:
        ...
