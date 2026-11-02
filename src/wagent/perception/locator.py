"""输入框自动定位：UIA → 视觉分隔线 → 比例兜底，三级降级 + 置信度。

为什么需要它：旧实现把输入框坐标写死在配置里（``INPUT_BOX_XY = (600, 800)``），
换电脑、改窗口大小、改 DPI 缩放就点空；点空之后还会继续执行「粘贴+回车」，
把内容打进别的窗口。

设计要点：
1. 不做「猜一次就信」：每个结果都带 ``method`` 与 ``confidence``；
2. 发送前还有一层校验（见 ``verify_clickable``），校验不过就拒绝发送；
3. 视觉法只依赖「输入框上方有一条横贯面板的分隔线」这一稳定布局特征，
   不依赖具体像素坐标。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wagent.config import LayoutSettings
from wagent.perception import regions
from wagent.perception.types import Box


@dataclass(frozen=True, slots=True)
class InputBox:
    """输入框定位结果。"""

    box: Box
    method: str  # "uia" | "visual" | "ratio"
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return int(self.box.cx), int(self.box.cy)

    def click_point(self) -> tuple[int, int]:
        """点在输入框上半部分，避免误触底部工具栏。"""
        x = int(self.box.cx)
        y = self.box.y1 + int(self.box.h * 0.45)
        return x, min(y, self.box.y2 - 1)


def locate_by_uia(hwnd: int, *, min_width: int = 120, min_height: int = 24) -> InputBox | None:
    """通过 UI Automation 控件树找输入框（微信 4.x 是 Qt，未必暴露，探到就最准）。

    只取「位于窗口下半部、面积最大的 Edit/Document 控件」。
    """
    try:
        import uiautomation as auto
    except Exception:  # noqa: BLE001 - 未安装/平台不支持则降级
        return None

    try:
        root = auto.ControlFromHandle(hwnd)
        window_rect = root.BoundingRectangle
        win_h = window_rect.bottom - window_rect.top
        if win_h <= 0:
            return None

        best = None
        stack = [root]
        visited = 0
        while stack and visited < 400:
            ctrl = stack.pop()
            visited += 1
            try:
                children = ctrl.GetChildren()
            except Exception:  # noqa: BLE001
                continue
            for child in children:
                stack.append(child)
                try:
                    name = child.ControlTypeName
                    rect = child.BoundingRectangle
                except Exception:  # noqa: BLE001
                    continue
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if name not in ("EditControl", "DocumentControl"):
                    continue
                if width < min_width or height < min_height:
                    continue
                # 只要窗口下半部
                if rect.top < window_rect.top + win_h * 0.45:
                    continue
                area = width * height
                if best is None or area > best[0]:
                    best = (area, rect)

        if best is None:
            return None
        rect = best[1]
        return InputBox(
            box=Box(rect.left, rect.top, rect.right, rect.bottom),
            method="uia",
            confidence=0.95,
        )
    except Exception:  # noqa: BLE001
        return None


def locate_by_visual(img: np.ndarray, layout: LayoutSettings) -> InputBox | None:
    """找输入框上方的横贯分隔线，反推输入框矩形。

    做法：在输入区候选范围（面板下部 45%）里按行计算「相邻行灰度差的平均绝对值」，
    取最强的水平线作为分隔线；再校验分隔线下方是否是大片均匀浅色区域。
    """
    import cv2

    h, w = img.shape[:2]
    area = regions.input_area(w, h, layout)
    sub = img[area.y1 : area.y2, area.x1 : area.x2]
    if sub.size == 0 or sub.shape[0] < 30:
        return None

    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.int16)
    row_diff = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    # 分隔线通常在输入区上部 15%~60% 之间
    lo = int(len(row_diff) * 0.10)
    hi = max(lo + 1, int(len(row_diff) * 0.60))
    idx = lo + int(np.argmax(row_diff[lo:hi]))
    strength = float(row_diff[idx])
    if strength < 3.0:  # 找不到明显的水平线
        return None

    top = area.y1 + idx + 3
    bottom = area.y2
    box_h = bottom - top
    if box_h < 30:
        return None

    # 校验：输入框文字区域应是大片均匀浅色
    probe_top = top + int(box_h * 0.10)
    probe_bottom = bottom - int(box_h * 0.25)
    probe = gray[probe_top - area.y1 : probe_bottom - area.y1, :]
    if probe.size == 0:
        return None
    uniformity = float(probe.std())
    if uniformity > 28.0:
        return None

    confidence = max(0.3, min(0.85, strength / 12.0)) * (1.0 - uniformity / 120.0)
    return InputBox(
        box=Box(area.x1, top, area.x2, bottom),
        method="visual",
        confidence=round(float(confidence), 3),
    )


def locate_by_ratio(img: np.ndarray, layout: LayoutSettings) -> InputBox:
    """最后兜底：面板中下部固定比例区域。"""
    h, w = img.shape[:2]
    panel = regions.chat_panel(w, h, layout)
    top = int(h * max(layout.message_area_bottom, 0.68))
    box = Box(int(panel.x1 + panel.w * 0.05), top, int(panel.x2 - panel.w * 0.05), int(h * 0.97))
    return InputBox(box=box, method="ratio", confidence=0.2)


def locate_input_box(
    img: np.ndarray,
    layout: LayoutSettings,
    *,
    hwnd: int | None = None,
) -> InputBox:
    """三级降级定位输入框。"""
    if hwnd is not None:
        uia = locate_by_uia(hwnd)
        if uia is not None:
            return uia
    visual = locate_by_visual(img, layout)
    if visual is not None:
        return visual
    return locate_by_ratio(img, layout)
