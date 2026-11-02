"""滚动 OCR 采史：不依赖任何导出/解密工具，直接把历史聊天"看"下来。

原理：聊天窗口向上滚动时，新一屏和上一屏会有重叠。对每一帧做消息序列比对，
找到「上一帧尾部 == 新帧头部」的最大重叠，把新增部分接上去，于是得到完整时间线。
连续两帧没有新增（说明已经滚到顶或滚不动了）就停止。

代价：慢（每帧都要 OCR + 等滚动动画），且依赖屏幕分辨率；换来的是零风险、可控。
适合夜间跑一个会话，配合 :class:`wagent.memory.store.MemoryStore` 建库。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wagent.memory.types import MessageRecord
from wagent.perception.tsutil import resolve_ts_text
from wagent.perception.types import Message


def _key(message: Message) -> tuple[str, str]:
    return (message.speaker.value, message.text.strip())


def merge_frames(
    existing: Sequence[Message], frame: Sequence[Message]
) -> tuple[list[Message], int, int]:
    """把新一帧拼到已有序列上，返回 ``(合并结果, 新增条数, 重叠条数)``。

    采史方向是**从最新一屏往上滚**，所以新帧里出现的是「更早的消息 + 与已有序列
    重叠的顶部」：新帧的**后缀**应当等于已有序列的**前缀**，把剩余部分插到最前面。
    （时间线始终按从旧到新存放。）

    重叠判定按消息序列（说话人 + 文本），不做模糊匹配——OCR 有错字时退化成
    "无重叠直接前置"，宁可多一条重复，也不要把两条不同的消息误合并。
    """
    base = list(existing)
    incoming = list(frame)
    if not incoming:
        return base, 0, 0
    if not base:
        return incoming, len(incoming), 0

    max_overlap = min(len(base), len(incoming))
    for k in range(max_overlap, 0, -1):
        if [_key(m) for m in incoming[-k:]] == [_key(m) for m in base[:k]]:
            return incoming[:-k] + base, len(incoming) - k, k
    return incoming + base, len(incoming), 0


@dataclass(slots=True)
class HarvestResult:
    messages: list[MessageRecord] = field(default_factory=list)
    frames: int = 0
    duplicates: int = 0
    stopped: str = ""  # max_frames | stagnant | empty

    def summary(self) -> str:
        return (
            f"采集 {self.frames} 帧，合并 {len(self.messages)} 条消息"
            f"（重叠 {self.duplicates} 条，停止原因：{self.stopped}）"
        )


@dataclass
class ScrollHarvester:
    """向上滚动并累积消息。"""

    grab_messages: Callable[[], Sequence[Message]]
    scroll_up: Callable[[], None]
    person: str = ""
    session: str = ""
    max_frames: int = 20
    max_stagnant: int = 2
    settle: Callable[[], None] = field(default=lambda: None)
    now: Callable[[], datetime] = field(default=lambda: datetime.now().astimezone())

    def harvest(self) -> HarvestResult:
        accumulated: list[Message] = []
        result = HarvestResult(stopped="max_frames")
        stagnant = 0

        for index in range(self.max_frames):
            frame = list(self.grab_messages())
            result.frames += 1
            if not frame:
                result.stopped = "empty"
                break

            accumulated, new_count, overlap = merge_frames(accumulated, frame)
            result.duplicates += overlap

            if new_count == 0:
                stagnant += 1
                if stagnant >= self.max_stagnant:
                    result.stopped = "stagnant"
                    break
            else:
                stagnant = 0

            if index == self.max_frames - 1:
                result.stopped = "max_frames"
                break

            self.scroll_up()
            self.settle()

        result.messages = self._to_records(accumulated)
        return result

    # ------------------------------------------------------------ 内部

    def _to_records(self, messages: Sequence[Message]) -> list[MessageRecord]:
        person = self.person or self.session or "unknown"
        base = self.now()
        records: list[MessageRecord] = []
        for index, message in enumerate(messages):
            ts = resolve_ts_text(message.ts_text, base) or (
                base + timedelta(seconds=index)
            ).isoformat(timespec="seconds")
            records.append(
                MessageRecord(
                    person=person,
                    speaker=message.speaker.value,
                    text=message.text,
                    ts=ts,
                    session=self.session or person,
                    source="scroll",
                )
            )
        return records
