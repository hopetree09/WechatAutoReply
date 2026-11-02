"""Windows 平台基础能力：DPI 感知、前台窗口、点击与键盘。

统一在进程启动时开启 Per-Monitor V2 DPI 感知，之后的截图坐标（物理像素）
和点击坐标（物理像素）才会一致——旧实现没有做这一步，缩放非 100% 时点击必偏。
"""

from __future__ import annotations

import contextlib
import ctypes
import time

_user32 = ctypes.WinDLL("user32", use_last_error=True)

_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


def enable_dpi_awareness() -> None:
    """开启 Per-Monitor V2 DPI 感知；已设置过会抛异常，忽略即可。"""
    try:
        _user32.SetProcessDpiAwarenessContext(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        return
    except Exception:
        pass
    try:
        # 旧系统兜底：shcore.SetProcessDpiAwareness(2)
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness(2)
    except Exception:
        with contextlib.suppress(Exception):
            _user32.SetProcessDPIAware()


def foreground_hwnd() -> int:
    """当前前台窗口句柄。"""
    return int(_user32.GetForegroundWindow())


def window_title(hwnd: int) -> str:
    """窗口标题（调试用）。"""
    import win32gui

    try:
        return win32gui.GetWindowText(hwnd)
    except Exception:
        return ""


def click(x: int, y: int, *, settle: float = 0.25) -> None:
    """在物理像素坐标处单击。"""
    import pyautogui

    pyautogui.PAUSE = 0.05
    pyautogui.click(int(x), int(y))
    time.sleep(settle)


def hotkey(*keys: str, settle: float = 0.1) -> None:
    import pyautogui

    pyautogui.PAUSE = 0.05
    pyautogui.hotkey(*keys)
    time.sleep(settle)


def press(key: str, *, settle: float = 0.1) -> None:
    import pyautogui

    pyautogui.PAUSE = 0.05
    pyautogui.press(key)
    time.sleep(settle)


def read_clipboard() -> str:
    import pyperclip

    try:
        return pyperclip.paste() or ""
    except Exception:
        return ""


def write_clipboard(text: str) -> None:
    import pyperclip

    pyperclip.copy(text)
