"""微信窗口定位。

微信 4.x 换了 UI 框架：主窗口类名是 ``Qt51514QWindowIcon``（Qt 版本号会变），
不再是 3.x 的 ``WeChatMainWndForPC``。旧代码用固定类名 ``FindWindow`` 在新版微信上
必然返回 0，所以这里改成：
1. 类名正则（``^Qt\\d+QWindowIcon$`` 等）+ 标题关键词；
2. 再用进程名（``Weixin.exe`` / ``WeChat.exe``）二次确认，避免误认别的 Qt 程序；
3. 主窗口被最小化时（Windows 会把窗口移到屏幕外，截不到内容）自动还原。

注意：微信进程会创建多个同进程窗口（输入法、托盘消息窗、隐藏辅助窗），
因此要按「标题含微信 + 尺寸合理」筛出主窗口。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from wagent.config import WeChatSettings


@dataclass(frozen=True, slots=True)
class WindowInfo:
    hwnd: int
    left: int
    top: int
    right: int
    bottom: int
    class_name: str
    title: str
    process_name: str = ""
    minimized: bool = False

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class WindowNotFoundError(RuntimeError):
    """找不到微信主窗口。"""


def _process_name(hwnd: int) -> str:
    try:
        import win32api
        import win32con
        import win32process

        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        try:
            return str(win32process.GetModuleFileNameEx(handle, 0)).split("\\")[-1]
        finally:
            win32api.CloseHandle(handle)
    except Exception:
        return ""


def list_windows() -> list[WindowInfo]:
    """枚举所有可见顶层窗口（调试/标定用）。"""
    import win32gui

    out: list[WindowInfo] = []

    def _cb(hwnd: int, _extra: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        except Exception:
            return True
        out.append(
            WindowInfo(
                hwnd=hwnd,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                class_name=win32gui.GetClassName(hwnd),
                title=title,
                process_name=_process_name(hwnd),
                minimized=bool(win32gui.IsIconic(hwnd)),
            )
        )
        return True

    win32gui.EnumWindows(_cb, None)
    return out


def _restore_minimized(hwnd: int) -> None:
    """还原最小化窗口（最小化时 GetWindowRect 返回屏幕外坐标，截不到内容）。"""
    import win32con
    import win32gui

    if not win32gui.IsIconic(hwnd):
        return
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    # 等窗口动画结束再取坐标
    for _ in range(20):
        if not win32gui.IsIconic(hwnd):
            break
        time.sleep(0.05)
    time.sleep(0.15)


def find_wechat_window(settings: WeChatSettings | None = None) -> WindowInfo:
    """定位微信主窗口；找不到时抛 :class:`WindowNotFoundError`。"""
    import win32gui

    settings = settings or WeChatSettings()
    class_re = re.compile(settings.window_class_regex)
    expected_procs = {p.lower() for p in settings.process_names}

    candidates: list[WindowInfo] = []
    minimzed_seen = False
    for win in list_windows():
        if not class_re.match(win.class_name):
            continue
        title_ok = any(k.lower() in win.title.lower() for k in settings.title_keywords)
        proc_ok = win.process_name.lower() in expected_procs
        if not (title_ok or proc_ok):
            continue
        if win.minimized:
            minimzed_seen = True
            if not settings.restore_minimized:
                continue
            _restore_minimized(win.hwnd)
            left, top, right, bottom = win32gui.GetWindowRect(win.hwnd)
            win = WindowInfo(
                hwnd=win.hwnd,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                class_name=win.class_name,
                title=win.title,
                process_name=win.process_name,
                minimized=False,
            )
        if win.width > 200 and win.height > 200:
            candidates.append(win)

    if not candidates:
        hint = "窗口已最小化且未开启自动还原（WAGENT_WECHAT__RESTORE_MINIMIZED=true）" if minimzed_seen else ""
        raise WindowNotFoundError(
            "未找到微信主窗口。请确认微信已登录且窗口未最小化"
            + (f"；当前状态：{hint}" if hint else "")
            + "；可用 scripts/probe_window.py 打印当前所有窗口类名。"
        )
    # 多开时取面积最大的那个
    return max(candidates, key=lambda w: w.width * w.height)
