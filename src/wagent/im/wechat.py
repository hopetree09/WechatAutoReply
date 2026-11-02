"""真实微信窗口适配器：把感知层各模块串成「一帧结构化消息」。

性能关键：聊天窗口绝大多数时间是完全静止的，所以
「消息区帧签名（哈希）不变 → 直接复用上一帧解析结果」，避免每秒都跑两次 OCR
（实测 1300x1366 窗口一次 tick 要 10s+，缓存命中后是毫秒级）。
窗口尺寸变化 / 每 30s 会刷新一次几何信息（输入框位置、消息区裁剪）。

注意这里只做**感知**，不发任何键鼠事件；执行由 :mod:`wagent.actuator` 负责。
"""

from __future__ import annotations

import time

import numpy as np

from wagent.config import Settings, get_settings
from wagent.im.base import Frame
from wagent.perception import regions
from wagent.perception.calibration import (
    BubblePalette,
    PaletteStore,
    bubble_extent,
    learn_palette,
)
from wagent.perception.capture import WindowCapturer
from wagent.perception.framesig import FrameCache
from wagent.perception.locator import InputBox, locate_by_ratio, locate_by_visual
from wagent.perception.ocr import ocr_region
from wagent.perception.parser import parse_ocr_lines, pick_session_name
from wagent.perception.speaker import classify_speaker, estimate_background_rgb
from wagent.perception.types import Box, Message
from wagent.perception.window import WindowInfo, find_wechat_window

GEOMETRY_TTL_S = 30.0


class WeChatAdapter:
    """微信（4.x）窗口适配器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.window: WindowInfo = find_wechat_window(self.settings.wechat)
        self.capturer = WindowCapturer(self.settings.capture)
        self.palette_store = PaletteStore(self.settings.calibration.store_path)
        self._palette: BubblePalette | None = None
        self._input: InputBox | None = None
        self._msg_rect: Box | None = None
        self._title_rect: Box | None = None
        self._geometry: tuple[int, int] | None = None
        self._geometry_at = 0.0
        self._cache: FrameCache[tuple[list[Message], str]] = FrameCache()
        self.last_image: np.ndarray | None = None
        self.last_cache_hit = False

    # ------------------------------------------------------------ 协议

    @property
    def hwnd(self) -> int:
        return self.window.hwnd

    def input_box(self) -> InputBox | None:
        return self._input

    @property
    def message_rect(self) -> Box | None:
        """最近一次计算出的消息区（采史时用来定位滚动中心）。"""
        return self._msg_rect

    def capture(self) -> Frame:
        img = self.capturer.grab(self.window.hwnd)
        self.last_image = img
        h, w = img.shape[:2]

        now = time.monotonic()
        if (
            self._geometry != (w, h)
            or self._msg_rect is None
            or now - self._geometry_at > GEOMETRY_TTL_S
        ):
            self._refresh_geometry(img)
            self._cache.invalidate()

        assert self._msg_rect is not None
        (messages, session), hit = self._cache.get_or_compute(
            img, self._msg_rect, lambda: self._parse(img)
        )
        self.last_cache_hit = hit
        return Frame(messages=list(messages), session=session, image=img)

    def close(self) -> None:
        self._palette = None
        self._input = None
        self._cache.invalidate()

    # ------------------------------------------------------------ 几何

    def _refresh_geometry(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        layout = self.settings.layout
        visual = locate_by_visual(img, layout)
        self._input = visual or locate_by_ratio(img, layout)

        msg_rect = regions.message_area(w, h, layout)
        if visual is not None and visual.box.y1 > msg_rect.y1 + 40:
            msg_rect = Box(msg_rect.x1, msg_rect.y1, msg_rect.x2, visual.box.y1 - 6)
        self._msg_rect = msg_rect
        self._title_rect = regions.title_area(w, h, layout)
        self._geometry = (w, h)
        self._geometry_at = time.monotonic()

    # ------------------------------------------------------------ 解析

    def _parse(self, img: np.ndarray) -> tuple[list[Message], str]:
        assert self._msg_rect is not None and self._title_rect is not None
        msg_rect = self._msg_rect
        title_lines = ocr_region(img, self._title_rect, self.settings.ocr)
        session = pick_session_name(title_lines)
        msg_lines = ocr_region(img, msg_rect, self.settings.ocr)

        background = estimate_background_rgb(img, msg_rect)
        palette = self._resolve_palette(img, msg_rect, msg_lines, background)

        def classify(box: Box) -> tuple:
            return classify_speaker(
                img,
                box,
                self.settings.bubble,
                background_rgb=background,
                my_bgr=palette.my_bgr if palette else None,
                other_bgr=palette.other_bgr if palette else None,
                content_left=msg_rect.x1,
                content_right=msg_rect.x2,
            )

        messages = parse_ocr_lines(
            msg_lines,
            classify,
            session=session,
            extent=lambda box: bubble_extent(img, box),
        )
        return messages, session

    def _resolve_palette(
        self,
        img: np.ndarray,
        msg_rect: Box,
        lines: list,
        background: np.ndarray | None,
    ) -> BubblePalette | None:
        if not self.settings.calibration.auto or background is None:
            return self._palette
        cached = self.palette_store.match(background)
        if cached is not None:
            self._palette = cached
            return cached
        learned = learn_palette(
            img,
            msg_rect,
            lines,
            panel_left=msg_rect.x1,
            panel_right=msg_rect.x2,
        )
        if learned is not None:
            self.palette_store.upsert(learned)
            self._palette = learned
        return self._palette
