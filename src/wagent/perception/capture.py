"""截图后端。

微信 4.x 是 Qt5 窗口（类名形如 ``Qt51514QWindowIcon``），实测
``PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT)`` 可以在**窗口被遮挡、不在前台**时
截到完整内容——这比旧实现依赖 dxcam（只能截可见屏幕，被遮挡就截到别的窗口）稳得多。

后端选择：
- ``printwindow``：默认，非侵入、抗遮挡；
- ``dxcam``：DDA 截可见屏幕，PrintWindow 失败时兜底；
- ``mss``：GDI 兜底（自绘窗口可能黑屏）。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import numpy as np

from wagent.config import CaptureSettings

_user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)  # type: ignore[attr-defined]

_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
_user32.GetWindowRect.restype = wintypes.BOOL
_user32.GetWindowDC.argtypes = [wintypes.HWND]
_user32.GetWindowDC.restype = wintypes.HDC
_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_user32.ReleaseDC.restype = ctypes.c_int
_user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
_user32.PrintWindow.restype = wintypes.BOOL

_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteObject.restype = wintypes.BOOL
_gdi32.DeleteDC.argtypes = [wintypes.HDC]
_gdi32.DeleteDC.restype = wintypes.BOOL
_gdi32.GetDIBits.argtypes = [
    wintypes.HDC,
    wintypes.HBITMAP,
    wintypes.UINT,
    wintypes.UINT,
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.UINT,
]
_gdi32.GetDIBits.restype = ctypes.c_int

PW_RENDERFULLCONTENT = 0x00000002
BI_RGB = 0
DIB_RGB_COLORS = 0


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class CaptureError(RuntimeError):
    """截图失败（窗口不存在 / 句柄无效 / 后端不可用）。"""


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise CaptureError(f"GetWindowRect 失败: hwnd={hwnd}")
    return rect.left, rect.top, rect.right, rect.bottom


def capture_printwindow(hwnd: int) -> np.ndarray:
    """用 PrintWindow 截取窗口内容，返回 BGR ``(H, W, 3)`` 数组。"""
    left, top, right, bottom = window_rect(hwnd)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise CaptureError(f"窗口尺寸非法: {width}x{height}")

    hdc_window = _user32.GetWindowDC(hwnd)
    if not hdc_window:
        raise CaptureError("GetWindowDC 失败（窗口可能已销毁）")
    hdc_mem = _gdi32.CreateCompatibleDC(hdc_window)
    hbitmap = _gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    old_obj = _gdi32.SelectObject(hdc_mem, hbitmap)

    try:
        ok = _user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)
        if not ok:
            raise CaptureError("PrintWindow 返回失败")

        header = _BITMAPINFOHEADER(
            biSize=ctypes.sizeof(_BITMAPINFOHEADER),
            biWidth=width,
            biHeight=-height,  # 负值 = 自上而下，省一次翻转
            biPlanes=1,
            biBitCount=32,
            biCompression=BI_RGB,
            biSizeImage=width * height * 4,
        )
        bmi = _BITMAPINFO(bmiHeader=header)
        buf = (ctypes.c_ubyte * (width * height * 4))()
        copied = _gdi32.GetDIBits(hdc_mem, hbitmap, 0, height, buf, ctypes.byref(bmi), DIB_RGB_COLORS)
        if copied == 0:
            raise CaptureError("GetDIBits 失败")

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(arr[:, :, :3])  # BGRA -> BGR
    finally:
        _gdi32.SelectObject(hdc_mem, old_obj)
        _gdi32.DeleteObject(hbitmap)
        _gdi32.DeleteDC(hdc_mem)
        _user32.ReleaseDC(hwnd, hdc_window)


class WindowCapturer:
    """按后端截取窗口；对瞬时失败做少量重试。"""

    def __init__(self, settings: CaptureSettings | None = None) -> None:
        self.settings = settings or CaptureSettings()
        self.backend = self.settings.backend
        self._cam = None
        self._mss = None
        if self.backend == "dxcam":
            import dxcam

            self._cam = dxcam.create(output_color="BGR")

    def grab(self, hwnd: int) -> np.ndarray:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                return self._grab_once(hwnd)
            except CaptureError as exc:  # PrintWindow 偶发失败，重试
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        raise CaptureError(f"连续 3 次截图失败: {last_err}")

    def _grab_once(self, hwnd: int) -> np.ndarray:
        if self.backend == "printwindow":
            img = capture_printwindow(hwnd)
            if img.std() < 1.0:  # 全黑/全白，视为失败
                raise CaptureError("PrintWindow 返回空白帧")
            return img
        if self.backend == "dxcam":
            left, top, right, bottom = window_rect(hwnd)
            frame = self._cam.grab(region=(left, top, right, bottom)) if self._cam else None
            if frame is None:
                raise CaptureError("dxcam 未返回帧")
            return frame
        return self._grab_mss()

    def _grab_mss(self) -> np.ndarray:
        import mss

        raise CaptureError("mss 后端需要窗口坐标，请改用 printwindow/dxcam")
