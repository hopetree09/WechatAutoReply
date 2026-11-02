"""模拟聊天通道：给单测 / CI / demo 用，不碰真实微信。

脚本化一组帧，按顺序返回；耗尽后重复最后一帧（模拟"画面停住"），
这样状态机不会因为反复喂同一帧而误触发。
"""

from __future__ import annotations

from collections.abc import Iterable

from wagent.im.base import Frame
from wagent.perception.locator import InputBox
from wagent.perception.types import Box


class MockAdapter:
    """按脚本返回帧的假适配器。"""

    def __init__(
        self,
        frames: Iterable[Frame],
        *,
        hwnd: int = 1,
        repeat_last: bool = True,
        click_point: tuple[int, int] = (100, 100),
    ) -> None:
        self.frames = list(frames)
        if not self.frames:
            raise ValueError("至少需要一帧")
        self._index = 0
        self._hwnd = hwnd
        self._repeat_last = repeat_last
        self._click_point = click_point
        self._input = InputBox(box=Box(10, 10, 300, 60), method="mock", confidence=1.0)

    @property
    def hwnd(self) -> int:
        return self._hwnd

    def capture(self) -> Frame:
        frame = self.frames[min(self._index, len(self.frames) - 1)]
        if self._index < len(self.frames) - 1:
            self._index += 1
        return frame

    def input_box(self) -> InputBox | None:
        return self._input

    @property
    def click_point(self) -> tuple[int, int]:
        return self._click_point

    def close(self) -> None:
        return None
