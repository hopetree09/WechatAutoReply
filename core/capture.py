"""截图模块：按微信窗口矩形返回 numpy 图像数组。"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Capturer:
    """截取微信窗口画面。

    微信 PC 版聊天内容自绘在 GPU 合成画布上，普通 GDI 截屏（mss/pyautogui）
    可能截到黑屏或空白，因此优先使用 dxcam（DDA，Desktop Duplication API）。
    """

    def __init__(self, backend: str = "dxcam") -> None:
        self.backend = backend
        self._cam = None
        if backend == "dxcam":
            import dxcam

            self._cam = dxcam.create(output_color="BGR")

    def grab_window(self, hwnd: int) -> Optional[np.ndarray]:
        """截取指定窗口，返回 BGR 数组 (H, W, 3)；窗口不可见时返回 None。"""
        import win32gui

        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return None

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)  # 物理像素
        if right <= left or bottom <= top:
            return None

        if self.backend == "dxcam":
            # DDA 截指定屏幕区域（物理像素）
            return self._cam.grab(region=(left, top, right, bottom))

        # 回退：mss（GDI，自绘窗口可能黑屏）
        import mss

        with mss.mss() as sct:
            mon = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            img = np.array(sct.grab(mon))[:, :, :3]  # BGRA -> BGR
            return img
