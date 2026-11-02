"""帧签名与解析缓存：没有变化就不重复跑 OCR。

OCR 一帧 250~800ms（大窗口更慢），而聊天窗口绝大多数时间是完全静止的。
这里用「消息区降采样后的哈希」作为廉价变化检测：没变就直接复用上一帧的解析结果，
让空闲轮询从「秒级」降到「毫秒级」；变化时才跑完整 OCR。

注意：光标闪烁/输入法候选框都在输入区，不在消息区，所以不会破坏签名稳定性。
"""

from __future__ import annotations

import hashlib
from typing import Generic, TypeVar

import numpy as np

from wagent.perception.types import Box

T = TypeVar("T")


def frame_signature(img: np.ndarray, region: Box, *, size: tuple[int, int] = (48, 48)) -> bytes:
    """计算指定区域的感知哈希（BLAKE2b 截断 16 字节）。"""
    h, w = img.shape[:2]
    r = region.clip(w, h)
    if r.w <= 0 or r.h <= 0:
        return b""
    sub = img[r.y1 : r.y2, r.x1 : r.x2]
    import cv2

    small = cv2.resize(sub, size, interpolation=cv2.INTER_AREA)
    # 量化到 4 级，抹掉 ±1 的渲染抖动
    quant = (small // 4).astype(np.uint8)
    return hashlib.blake2b(quant.tobytes(), digest_size=16).digest()


class FrameCache(Generic[T]):
    """按帧签名缓存解析结果。"""

    def __init__(self) -> None:
        self._signature: bytes | None = None
        self._value: T | None = None
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, img: np.ndarray, region: Box, compute) -> tuple[T, bool]:
        """返回 ``(解析结果, 是否命中缓存)``。"""
        signature = frame_signature(img, region)
        if signature and signature == self._signature and self._value is not None:
            self.hits += 1
            return self._value, True
        value = compute()
        self._signature = signature
        self._value = value
        self.misses += 1
        return value, False

    def invalidate(self) -> None:
        self._signature = None
        self._value = None
