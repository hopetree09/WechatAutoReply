"""执行层：把「回复文本」安全地敲进微信输入框。

旧实现的三个风险：
1. 只有 ``AUTO_SEND`` 布尔开关，一旦打开就盲发：不校验前台窗口、不校验焦点，
   微信被遮挡时会把回复粘到当前前台程序（IDE/终端/别的聊天框）并回车；
2. 剪贴板被永久改写（贴的是最后一片回复），用户的复制内容丢失；
3. 所谓「打字速度 ∝ 消息长度」是死代码——``estimate_typing_time`` 从未被调用。

新实现：
- 三级模式 ``dry_run / review / send``，默认 dry_run 只出计划；
- 点击前后各校验一次前台窗口，不一致立即中止；
- 剪贴板保存/恢复（异常也恢复）；
- 真正把每片文字的耗时计入等待，长消息就是慢、短消息就是快；
- 所有系统调用通过 :class:`InputBackend` 注入，便于单测与录制回放。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum


class SendMode(StrEnum):
    DRY_RUN = "dry_run"  # 只生成计划，不碰键鼠
    REVIEW = "review"  # 需要人工确认
    SEND = "send"  # 直接发送


@dataclass(frozen=True, slots=True)
class TypingProfile:
    """打字节奏参数。"""

    cps_base: float = 0.30  # 每字基础秒数
    cps_jitter: float = 0.25  # 逐字抖动 ±25%
    chunk_size: tuple[int, int] = (12, 22)  # 分片长度范围（字）
    pause_between_chunks: tuple[float, float] = (0.8, 2.5)  # 分片间停顿（秒）
    think_delay: tuple[float, float] = (1.0, 4.0)  # 收到消息后的"思考"停顿


def estimate_typing_time(
    text: str, profile: TypingProfile, *, rng: random.Random | None = None
) -> float:
    """估算「假装打字」总时长（秒）：字数 × 每字耗时 × 抖动 + 分片停顿。"""
    if not text:
        return 0.0
    rng = rng or random.Random()
    total = sum(
        profile.cps_base * rng.uniform(1 - profile.cps_jitter, 1 + profile.cps_jitter)
        for _ in text
    )
    chunk = rng.randint(*profile.chunk_size)
    pauses = max(0, (len(text) - 1) // chunk)
    total += sum(rng.uniform(*profile.pause_between_chunks) for _ in range(pauses))
    return total


@dataclass(frozen=True, slots=True)
class InputBackend:
    """系统能力集合（默认接 winapi/pyautogui，测试时注入假实现）。"""

    click: Callable[[int, int], None]
    press: Callable[[str], None]
    hotkey: Callable[[Sequence[str]], None]
    write: Callable[[str, float], None]
    clipboard_read: Callable[[], str]
    clipboard_write: Callable[[str], None]
    foreground_hwnd: Callable[[], int]
    sleep: Callable[[float], None]


def default_backend() -> InputBackend:
    """Windows 默认后端。"""
    import pyautogui

    from wagent import winapi

    return InputBackend(
        click=winapi.click,
        press=winapi.press,
        hotkey=lambda keys: winapi.hotkey(*keys),
        write=lambda text, interval: pyautogui.write(text, interval=interval),
        clipboard_read=winapi.read_clipboard,
        clipboard_write=winapi.write_clipboard,
        foreground_hwnd=winapi.foreground_hwnd,
        sleep=time.sleep,
    )


@dataclass(frozen=True, slots=True)
class SendPlan:
    """一次发送的计划（dry_run 模式下就是最终产物）。"""

    text: str
    estimated_typing_s: float
    think_delay_s: float
    chunks: tuple[str, ...]

    @property
    def total_s(self) -> float:
        return self.think_delay_s + self.estimated_typing_s


@dataclass(frozen=True, slots=True)
class SendResult:
    sent: bool
    reason: str
    elapsed_s: float
    plan: SendPlan


def _is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


class SafeSender:
    """带安全校验的拟人化发送器。"""

    def __init__(
        self,
        profile: TypingProfile | None = None,
        *,
        backend: InputBackend | None = None,
        send_key: str = "enter",
        rng: random.Random | None = None,
        confirm: Callable[[str], bool] | None = None,
        require_foreground: bool = True,
        restore_clipboard: bool = True,
    ) -> None:
        self.profile = profile or TypingProfile()
        self._backend = backend
        self.send_key = send_key
        self.rng = rng or random.Random()
        self.confirm = confirm
        self.require_foreground = require_foreground
        self.restore_clipboard = restore_clipboard

    @property
    def backend(self) -> InputBackend:
        """懒加载系统后端：dry_run 模式完全不会碰 pyautogui/pywin32。"""
        if self._backend is None:
            self._backend = default_backend()
        return self._backend

    # ------------------------------------------------------------ 计划

    def plan(self, text: str) -> SendPlan:
        text = text.strip()
        chunk = self.rng.randint(*self.profile.chunk_size)
        chunks = tuple(text[i : i + chunk] for i in range(0, len(text), chunk)) if text else ()
        typing = estimate_typing_time(text, self.profile, rng=self.rng)
        think = self.rng.uniform(*self.profile.think_delay)
        return SendPlan(
            text=text,
            estimated_typing_s=round(typing, 3),
            think_delay_s=round(think, 3),
            chunks=chunks,
        )

    # ------------------------------------------------------------ 发送

    def send(
        self,
        text: str,
        *,
        target_hwnd: int,
        click_point: tuple[int, int],
        mode: SendMode = SendMode.DRY_RUN,
        confirm: Callable[[str], bool] | None = None,
    ) -> SendResult:
        started = time.monotonic()
        plan = self.plan(text)
        if not plan.text:
            return SendResult(False, "empty_text", 0.0, plan)

        if mode is SendMode.DRY_RUN:
            return SendResult(False, "dry_run", 0.0, plan)

        if mode is SendMode.REVIEW:
            checker = confirm or self.confirm
            if checker is None or not checker(plan.text):
                return SendResult(False, "review_rejected", 0.0, plan)

        if self.require_foreground and self.backend.foreground_hwnd() != target_hwnd:
            return SendResult(False, "not_foreground", time.monotonic() - started, plan)

        self.backend.sleep(plan.think_delay_s)
        self.backend.click(*click_point)
        self.backend.sleep(0.25)

        # 点击后再校验一次：若微信被别的窗口盖住，这次点击可能已经把焦点给了别的程序
        if self.require_foreground and self.backend.foreground_hwnd() != target_hwnd:
            return SendResult(False, "focus_lost", time.monotonic() - started, plan)

        backup = self.backend.clipboard_read() if self.restore_clipboard else None
        try:
            self._type(plan)
        finally:
            if self.restore_clipboard and backup is not None:
                self.backend.clipboard_write(backup)

        self.backend.sleep(self.rng.uniform(0.3, 0.8))
        if self.send_key == "ctrl+enter":
            self.backend.hotkey(("ctrl", "enter"))
        else:
            self.backend.press("enter")

        return SendResult(True, "sent", round(time.monotonic() - started, 3), plan)

    # ------------------------------------------------------------ 内部

    def _type(self, plan: SendPlan) -> None:
        """逐片输入并等待「本应花掉的打字时间」，让长消息自然变慢。"""
        for index, chunk in enumerate(plan.chunks):
            if _is_ascii(chunk):
                self.backend.write(chunk, self.rng.uniform(0.04, 0.12))
            else:
                self.backend.clipboard_write(chunk)
                self.backend.hotkey(("ctrl", "v"))
            self.backend.sleep(self._chunk_delay(chunk))
            if index < len(plan.chunks) - 1:
                self.backend.sleep(self.rng.uniform(*self.profile.pause_between_chunks))

    def _chunk_delay(self, chunk: str) -> float:
        return sum(
            self.profile.cps_base * self.rng.uniform(1 - self.profile.cps_jitter, 1 + self.profile.cps_jitter)
            for _ in chunk
        )
