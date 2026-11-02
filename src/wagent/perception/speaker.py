"""说话人判别：气泡底色「三分类最近邻 + 加权投票」+ 左右对齐兜底。

旧实现的两个致命问题：
1. 只采样文本框左侧一个固定窗口，采到聊天区浅灰背景时算出的色距（约 26~31）
   仍小于容差 40，于是背景被当成「对方的白气泡」；
2. 采样失败直接默认 ``other``，导致程序倾向于把任何文字都当成对方说的话。

新实现：
- **背景色自动估计**：消息区出现频率最高的颜色就是聊天背景，不需要手工标定；
- **三分类**：每个采样块归到 {我的气泡色, 对方气泡色, 背景色} 中色距最近的一类；
- **加权投票**：气泡上/下内边距（紧贴文字、必然在气泡内）权重 2，左右权重 1；
- 背景票获胜 → ``UNKNOWN``（上层因此不会触发回复），而不是瞎猜一个。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from wagent.config import BubbleSettings
from wagent.perception.types import Box, Speaker

_MY = 0
_OTHER = 1
_BG = 2
_CLASS_TO_SPEAKER = {_MY: Speaker.ME, _OTHER: Speaker.OTHER}


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.array(rgb[::-1], dtype=float)


def _patch_mean(img: np.ndarray, box: Box) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1 = max(0, box.x1), max(0, box.y1)
    x2, y2 = min(w, box.x2), min(h, box.y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi.reshape(-1, 3).mean(axis=0)


def estimate_background_rgb(img: np.ndarray, region: Box | None = None) -> np.ndarray | None:
    """估计聊天区背景色：对颜色做粗量化后取众数。

    返回 BGR 颜色；区域太小时返回 ``None``。
    """
    if img is None or img.size == 0:
        return None
    if region is not None:
        h, w = img.shape[:2]
        r = region.clip(w, h)
        if r.w <= 0 or r.h <= 0:
            return None
        img = img[r.y1 : r.y2, r.x1 : r.x2]

    # 缩小采样，避免大图统计过慢；聊天背景是纯色，直接用精确颜色取众数
    sample = img[:: max(1, img.shape[0] // 160), :: max(1, img.shape[1] // 160)]
    counter = Counter(map(tuple, sample.reshape(-1, 3).tolist()))
    (b, g, r), _count = counter.most_common(1)[0]
    return np.array([b, g, r], dtype=float)


def _alignment_vote(box: Box, content_left: int, content_right: int) -> Speaker | None:
    """气泡水平位置兜底：右侧贴边 = 我，左侧贴边 = 对方。"""
    if content_right <= content_left:
        return None
    span = float(content_right - content_left)
    right_ratio = (box.x2 - content_left) / span
    left_ratio = (box.x1 - content_left) / span
    if right_ratio >= 0.82:
        return Speaker.ME
    if left_ratio <= 0.18:
        return Speaker.OTHER
    return None


def _sample_patches(box: Box, min_px: int) -> list[tuple[Box, int]]:
    """构造采样块：上/下内边距（权重 2）+ 左/右（权重 1）。"""
    patches: list[tuple[Box, int]] = []
    # 气泡上下内边距：文字框上方 6px、下方 6px，横向与文字对齐
    patches.append((Box(box.x1, box.y1 - min_px - 3, box.x2, box.y1 - min_px + 0 + 2), 2))
    patches.append((Box(box.x1, box.y2 + 1, box.x2, box.y2 + 1 + min_px + 2), 2))
    # 左右：贴近文字，气泡内边距足够时能采到气泡色
    patches.append((Box(box.x1 - 14, box.y1, box.x1 - 3, box.y2), 1))
    patches.append((Box(box.x2 + 3, box.y1, box.x2 + 14, box.y2), 1))
    return patches


def classify_speaker(
    img: np.ndarray,
    box: Box,
    settings: BubbleSettings,
    *,
    background_rgb: np.ndarray | tuple[int, int, int] | None = None,
    my_bgr: tuple[int, int, int] | None = None,
    other_bgr: tuple[int, int, int] | None = None,
    content_left: int = 0,
    content_right: int = 0,
) -> tuple[Speaker, float]:
    """返回 ``(说话人, 置信度)``；无法判定时返回 ``(UNKNOWN, 0.0)``。

    ``my_bgr`` / ``other_bgr`` 用于传入自动标定学到的气泡色（覆盖配置里的默认值）。
    """
    h, w = img.shape[:2]
    box = box.clip(w, h)
    if box.w <= 0 or box.h <= 0:
        return Speaker.UNKNOWN, 0.0

    bg = (
        np.asarray(background_rgb, dtype=float)
        if background_rgb is not None
        else estimate_background_rgb(img)
    )
    if bg is None:
        bg = np.array(img.reshape(-1, 3).mean(axis=0), dtype=float)

    center_my = (
        np.asarray(my_bgr, dtype=float) if my_bgr is not None else _rgb_to_bgr(settings.my_rgb)
    )
    center_other = (
        np.asarray(other_bgr, dtype=float)
        if other_bgr is not None
        else _rgb_to_bgr(settings.other_rgb)
    )
    centers = [center_my, center_other, bg]
    votes = {_MY: 0, _OTHER: 0, _BG: 0}

    for patch, weight in _sample_patches(box, settings.min_patch_px):
        mean = _patch_mean(img, patch)
        if mean is None:
            continue
        distances = [_distance(mean, c) for c in centers]
        nearest = int(np.argmin(distances))
        if distances[nearest] > settings.tolerance:
            continue  # 采到壁纸/图片等未知颜色，这一票作废
        votes[nearest] += weight

    total = sum(votes.values())
    if total == 0:
        return Speaker.UNKNOWN, 0.0

    # 背景票过半 → 这行文字不在气泡里（系统提示/链接卡片/说明文字）
    if votes[_BG] >= max(votes[_MY], votes[_OTHER]):
        return Speaker.UNKNOWN, 0.0

    speaker = Speaker.ME if votes[_MY] > votes[_OTHER] else Speaker.OTHER
    confidence = max(votes[_MY], votes[_OTHER]) / total
    if confidence < settings.min_vote_ratio:
        # 我/对方票数接近时，用水平对齐兜底
        fallback = _alignment_vote(box, content_left, content_right)
        if fallback is None:
            return Speaker.UNKNOWN, 0.0
        return fallback, round(confidence, 3)
    return speaker, round(confidence, 3)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
