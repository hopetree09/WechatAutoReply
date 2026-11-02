"""安全执行层单测：用假 InputBackend 验证模式、前台校验、剪贴板保护与打字时长。"""

from __future__ import annotations

import contextlib
import random

from wagent.actuator import (
    InputBackend,
    SafeSender,
    SendMode,
    TypingProfile,
    estimate_typing_time,
)

PROFILE = TypingProfile(
    cps_base=0.10,
    cps_jitter=0.0,
    chunk_size=(5, 5),
    pause_between_chunks=(0.5, 0.5),
    think_delay=(1.0, 1.0),
)

HWND = 12345
CLICK_POINT = (100, 200)


class FakeWin:
    """记录所有系统调用，模拟前台窗口切换。"""

    def __init__(self, *, start_clipboard: str = "用户原来的剪贴板") -> None:
        self.events: list[tuple] = []
        self.sleeps: list[float] = []
        self.clipboard = start_clipboard
        self.foreground_queue: list[int] = []

    def backend(self) -> InputBackend:
        return InputBackend(
            click=self._click,
            press=lambda key: self.events.append(("press", key)),
            hotkey=lambda keys: self.events.append(("hotkey", tuple(keys))),
            write=lambda text, interval: self.events.append(("write", text)),
            clipboard_read=self._clip_read,
            clipboard_write=self._clip_write,
            foreground_hwnd=self._foreground,
            sleep=self.sleeps.append,
        )

    def _click(self, x: int, y: int) -> None:
        self.events.append(("click", x, y))

    def _clip_read(self) -> str:
        return self.clipboard

    def _clip_write(self, text: str) -> None:
        self.clipboard = text
        self.events.append(("clipboard", text))

    def _foreground(self) -> int:
        return self.foreground_queue.pop(0) if self.foreground_queue else HWND


def make_sender(win: FakeWin, **kwargs) -> SafeSender:
    kwargs.setdefault("rng", random.Random(42))
    return SafeSender(PROFILE, backend=win.backend(), **kwargs)


# ---------------------------------------------------------------- 模式


def test_dry_run_touches_nothing() -> None:
    win = FakeWin()
    result = make_sender(win).send("你好呀", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.DRY_RUN)
    assert result.sent is False
    assert result.reason == "dry_run"
    assert result.plan.estimated_typing_s > 0
    assert win.events == []
    assert win.sleeps == []


def test_empty_text_rejected() -> None:
    win = FakeWin()
    result = make_sender(win).send("   ", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert result.sent is False
    assert result.reason == "empty_text"


def test_review_without_confirm_is_rejected() -> None:
    win = FakeWin()
    result = make_sender(win).send("你好", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.REVIEW)
    assert result.sent is False
    assert result.reason == "review_rejected"
    assert win.events == []


def test_review_with_confirm_sends() -> None:
    win = FakeWin()
    sender = make_sender(win, confirm=lambda text: text == "你好")
    result = sender.send("你好", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.REVIEW)
    assert result.sent is True
    assert ("click", *CLICK_POINT) in win.events


# ---------------------------------------------------------------- 前台校验


def test_aborts_when_wechat_not_foreground() -> None:
    win = FakeWin()
    win.foreground_queue = [999]  # 前台是别的程序
    result = make_sender(win).send("你好", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert result.sent is False
    assert result.reason == "not_foreground"
    assert win.events == []  # 一次点击都没有发生


def test_aborts_if_focus_lost_after_click() -> None:
    win = FakeWin()
    win.foreground_queue = [HWND, 999]  # 点击前是微信，点击后焦点跑到别的程序
    result = make_sender(win).send("你好", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert result.sent is False
    assert result.reason == "focus_lost"
    # 只发生了点击，没有输入任何内容
    assert [e[0] for e in win.events] == ["click"]


def test_foreground_check_can_be_disabled() -> None:
    win = FakeWin()
    win.foreground_queue = [999, 999]
    sender = make_sender(win, require_foreground=False)
    result = sender.send("你好", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert result.sent is True


# ---------------------------------------------------------------- 剪贴板 / 输入


def test_clipboard_is_restored() -> None:
    win = FakeWin(start_clipboard="用户原来的剪贴板")
    result = make_sender(win).send("这是中文回复", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert result.sent is True
    assert win.clipboard == "用户原来的剪贴板"
    # 中间确实发生过粘贴
    assert any(e[0] == "hotkey" and e[1] == ("ctrl", "v") for e in win.events)


def test_clipboard_restored_even_on_error() -> None:
    win = FakeWin(start_clipboard="原始内容")

    def boom(_text: str) -> None:
        raise RuntimeError("输入失败")

    backend = win.backend()
    sender = SafeSender(
        PROFILE,
        backend=InputBackend(
            click=backend.click,
            press=backend.press,
            hotkey=lambda _keys: boom("x"),
            write=backend.write,
            clipboard_read=backend.clipboard_read,
            clipboard_write=backend.clipboard_write,
            foreground_hwnd=backend.foreground_hwnd,
            sleep=backend.sleep,
        ),
        rng=random.Random(42),
    )
    with contextlib.suppress(RuntimeError):
        sender.send("中文", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert win.clipboard == "原始内容"


def test_ascii_uses_direct_typing() -> None:
    win = FakeWin()
    make_sender(win).send("ok great", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert any(e[0] == "write" for e in win.events)
    # 纯 ASCII 不经过剪贴板粘贴（唯一的剪贴板事件是最后的恢复）
    assert not any(e[0] == "hotkey" and e[1] == ("ctrl", "v") for e in win.events)


def test_send_key_variants() -> None:
    win = FakeWin()
    make_sender(win).send("好的", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert ("press", "enter") in win.events

    win2 = FakeWin()
    make_sender(win2, send_key="ctrl+enter").send(
        "好的", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND
    )
    assert ("hotkey", ("ctrl", "enter")) in win2.events


# ---------------------------------------------------------------- 打字时长


def test_typing_time_grows_with_length() -> None:
    win_short, win_long = FakeWin(), FakeWin()
    make_sender(win_short).send("好的", target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    long_text = "今天下午三点在楼下咖啡店见吧，我把简历打印一份带过去" * 2
    make_sender(win_long).send(long_text, target_hwnd=HWND, click_point=CLICK_POINT, mode=SendMode.SEND)
    assert sum(win_long.sleeps) > sum(win_short.sleeps) * 3


def test_estimate_typing_time_reasonable() -> None:
    profile = TypingProfile(cps_base=0.3, cps_jitter=0.0, chunk_size=(10, 10),
                            pause_between_chunks=(1.0, 1.0), think_delay=(0.0, 0.0))
    # 5 字：1.5s；20 字：6s + 1 次停顿 = 7s
    assert abs(estimate_typing_time("哈哈可以啊", profile, rng=random.Random(0)) - 1.5) < 0.01
    assert abs(estimate_typing_time("一" * 20, profile, rng=random.Random(0)) - 7.0) < 0.01
