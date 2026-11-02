"""感知层数据结构。

设计要点：
- ``Box`` 统一为「左上/右下」轴对齐矩形，内部坐标是**相对窗口截图**的像素；
  OCR 的四点框在解析入口一次性转成 ``Box``。
- ``Message`` 带 ``fingerprint``：对话去重不再用裸 ``(speaker, text)``，
  而是「会话 + 归一化文本 + 行位置分桶」，避免同一句话第二次发送被永久吞掉。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Speaker(str, Enum):
    ME = "me"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Box:
    """轴对齐矩形（像素，原点在截图左上角）。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_quad(cls, quad: object) -> Box:
        """把 OCR 的四点框 ``[[x,y], ...]`` 转成外接矩形。"""
        pts = np.asarray(quad, dtype=float).reshape(-1, 2)
        return cls(
            x1=int(pts[:, 0].min()),
            y1=int(pts[:, 1].min()),
            x2=int(pts[:, 0].max()),
            y2=int(pts[:, 1].max()),
        )

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def expand(self, dx: int, dy: int) -> Box:
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, width: int, height: int) -> Box:
        return Box(
            max(0, min(width, self.x1)),
            max(0, min(height, self.y1)),
            max(0, min(width, self.x2)),
            max(0, min(height, self.y2)),
        )


@dataclass(frozen=True, slots=True)
class OcrLine:
    """一行 OCR 结果。"""

    text: str
    score: float
    box: Box


@dataclass(frozen=True, slots=True)
class Message:
    """一条结构化聊天消息。"""

    speaker: Speaker
    text: str
    box: Box
    confidence: float = 1.0
    session: str = ""
    # 行位置分桶：用于去重时区分「同一句话在不同时间/位置再次出现」
    row_bucket: int = 0

    @property
    def fingerprint(self) -> str:
        raw = f"{self.session}|{self.speaker.value}|{self.text.strip()}|{self.row_bucket}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def row_bucket(y: float, bucket_height: float = 48.0) -> int:
    """把 y 坐标离散成桶，避免 1~2 像素抖动导致指纹变化。"""
    return int(y // bucket_height)
