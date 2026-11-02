"""触发状态机：决定「什么时候该回、回哪一条、不重复回」。

旧实现（``main.py``）只有一行判断：``msgs[-1].speaker == "other" and conv.update(msgs)``，
由此引出四个实际故障：
1. **启动即插话**：第一次循环所有消息都算"新消息"，会直接回复屏幕上几小时前的旧消息；
2. **重复内容永久失效**：去重键是 ``(speaker, text)`` 且 ``_seen`` 永不清理，
   对方第二次发「哈哈哈」永远不会再触发；
3. **画面抖动误触发**：OCR 结果只要多出一条就算"有变化"，滚动/闪烁都能触发；
4. **会话串台**：切换聊天窗口后上下文混在一起。

这里用显式状态机 + 「对话尾部指纹」解决：

    BASELINE ──首帧──▶ IDLE ──对方新消息──▶ REPLYING ──回复出现在屏幕上──▶ COOLDOWN ──▶ IDLE
                          ▲                                   │
                          └──────────── skip/fail/超时 ◀──────┘

判定「新消息」不比历史指纹，而是比**最近 tail_len 条消息的尾部序列**：
只有尾部变化、且最新一条是对方、且它和上一帧的尾部末条不同，才算新消息。
这样滚动（旧消息上移、尾部不变）不会触发，重复内容（中间夹着自己的回复）仍会触发。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from wagent.perception.types import Message, Speaker


class Phase(StrEnum):
    BASELINE = "baseline"  # 启动基线：只记录，不回复
    IDLE = "idle"  # 等待对方新消息
    REPLYING = "replying"  # 已决定回复，等待回复出现在屏幕上
    COOLDOWN = "cooldown"  # 回复后的冷却期，避免连发


@dataclass(frozen=True, slots=True)
class Trigger:
    """一次「该回复了」的决策结果。"""

    message: Message
    context: list[Message]
    session: str


Tail = tuple[tuple[str, str], ...]


def _tail_of(messages: Sequence[Message], length: int) -> Tail:
    return tuple((m.speaker.value, m.text.strip()) for m in messages[-length:])


def _is_appended(prev: Tail, new: Tail) -> bool:
    """判断新尾部是否在旧尾部之后「又出现了新消息」。

    只看尾部序列容易踩两个坑：轮询间隔内对方连发多条（尾部一次跳好几条）、
    以及 tail_len 截断导致旧尾部整体移出。所以判据是：
    **上一帧的最后一条消息仍出现在这一帧的「非末条」序列里** → 说明末尾是新增的。

    - 新消息到达（追加 1 条或多条）→ True；
    - 滚动导致整体上移、末尾没新增 → 上一帧末条消失 → False；
    - 中间消息 OCR 抖动 → 上一帧末条消失 → False；
    - 对方连发两条一模一样的消息 → 上一帧末条仍在 → True。
    """
    if not new:
        return False
    if not prev:
        return True
    body = new[:-1]
    return any(item == prev[-1] for item in body)


@dataclass
class ConversationState:
    """单个会话的触发状态机。

    用法::

        state = ConversationState()
        trigger = state.observe(messages, session=session)
        if trigger:
            ... 生成回复 ...
            state.mark_sent(reply_text)   # 或 state.skip() / state.fail()
    """

    tail_len: int = 4
    context_len: int = 12
    cooldown_s: float = 8.0
    reply_timeout_s: float = 20.0
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    session: str = ""
    phase: Phase = Phase.BASELINE
    tail: Tail = ()
    baseline_done: bool = False
    # 统计（便于观测/调参）
    triggers: int = 0
    skips: int = 0
    _cooldown_until: float = 0.0
    _reply_started_at: float = 0.0
    _reply_text: str = ""
    _pending: Message | None = None

    # ------------------------------------------------------------ 对外接口

    def observe(self, messages: Sequence[Message], *, session: str = "") -> Trigger | None:
        """喂入一帧消息，返回触发结果（``None`` 表示不需要回复）。"""
        messages = [m for m in messages if m.text.strip()]

        # 会话切换：重新建立基线，绝不回复另一个会话里的历史消息
        if session and session != self.session:
            self.reset(session)

        new_tail = _tail_of(messages, self.tail_len)

        # 第一帧：建立基线
        if not self.baseline_done:
            self.tail = new_tail
            self.baseline_done = True
            self.phase = Phase.IDLE
            if session:
                self.session = session
            return None

        # 回复中：等自己的回复出现在屏幕最下方；超时则允许重试
        if self.phase is Phase.REPLYING:
            self.tail = new_tail
            if new_tail and new_tail[-1][0] == Speaker.ME.value:
                self.phase = Phase.COOLDOWN
                self._cooldown_until = self._clock() + self.cooldown_s
            elif self._clock() - self._reply_started_at > self.reply_timeout_s:
                self.fail()
            return None

        # 尾部没变：画面静止（滚动/OCR 抖动都归到这里）
        if new_tail == self.tail:
            return None

        prev_tail, self.tail = self.tail, new_tail

        if self.phase is Phase.COOLDOWN:
            if self._clock() < self._cooldown_until:
                return None
            self.phase = Phase.IDLE

        bottom = messages[-1] if messages else None
        if bottom is None or bottom.speaker is not Speaker.OTHER:
            return None

        # 只有「末尾追加」才算新消息；滚动/中间抖动/重绘一律不触发
        if not _is_appended(prev_tail, new_tail):
            return None

        self._pending = bottom
        self.phase = Phase.REPLYING
        self._reply_started_at = self._clock()
        self._reply_text = ""
        self.triggers += 1
        return Trigger(
            message=bottom,
            context=list(messages[-self.context_len :]),
            session=self.session,
        )

    def mark_sent(self, reply_text: str) -> None:
        """执行层已发送：开始等待回复出现在屏幕上。"""
        self._reply_text = reply_text
        self._reply_started_at = self._clock()
        self.phase = Phase.REPLYING

    def skip(self) -> None:
        """本次不回复（例如意图判定为「无需回复」）。"""
        self._pending = None
        self._reply_text = ""
        self.phase = Phase.IDLE
        self.skips += 1

    def fail(self) -> None:
        """执行失败/超时：回到 IDLE，并清空尾部指纹以便下一帧重新评估同一条消息。"""
        self._pending = None
        self._reply_text = ""
        self.phase = Phase.IDLE
        self.tail = ()

    def reset(self, session: str = "") -> None:
        """切换会话或手动重置：回到基线状态。"""
        self.session = session
        self.phase = Phase.BASELINE
        self.tail = ()
        self.baseline_done = False
        self._pending = None
        self._reply_text = ""
        self._cooldown_until = 0.0

    # ------------------------------------------------------------ 只读属性

    @property
    def pending(self) -> Message | None:
        return self._pending

    @property
    def reply_text(self) -> str:
        return self._reply_text

    def snapshot(self) -> dict:
        """给日志/调试用的状态快照。"""
        return {
            "session": self.session,
            "phase": self.phase.value,
            "baseline_done": self.baseline_done,
            "tail": self.tail,
            "triggers": self.triggers,
            "skips": self.skips,
        }
