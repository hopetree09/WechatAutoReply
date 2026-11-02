"""气泡配色自动标定（自举学习），让项目不依赖手工配置颜色。

背景：微信 4.x 改版后气泡绿实际是 RGB(157, 242, 159)（旧版 3.x 是 149/236/105），
写死颜色必然失效；深色主题、自定义主题就更不用说了。

做法（几何引导自举）：
1. 用 OCR 文本框的左右留白判断这条消息大概属于哪一侧：
   - 贴近聊天面板右边界 → 「我」（短消息尤其明显）；
   - 贴近左边界 → 「对方」；
   - 两侧留白接近（长消息）→ 跳过，不参与学习。
2. 采样这些消息气泡的上/下/左/右内边距颜色，按侧别取中位数 → 得到两种气泡色；
3. 背景色取消息区众数；
4. 结果按「背景色」作为主题指纹缓存，换主题时自动重新学习。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wagent.perception.speaker import _patch_mean, _sample_patches, estimate_background_rgb
from wagent.perception.types import Box, OcrLine


@dataclass(frozen=True, slots=True)
class BubblePalette:
    """一套主题下的气泡配色（BGR）。"""

    background_bgr: tuple[int, int, int]
    my_bgr: tuple[int, int, int] | None = None
    other_bgr: tuple[int, int, int] | None = None

    @property
    def theme_key(self) -> str:
        r, g, b = self.background_bgr
        return f"#{r:02x}{g:02x}{b:02x}"

    def to_dict(self) -> dict:
        return {
            "background_bgr": list(self.background_bgr),
            "my_bgr": list(self.my_bgr) if self.my_bgr else None,
            "other_bgr": list(self.other_bgr) if self.other_bgr else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BubblePalette:
        def _tup(value):
            return None if value is None else (int(value[0]), int(value[1]), int(value[2]))

        return cls(
            background_bgr=_tup(data["background_bgr"]),  # type: ignore[arg-type]
            my_bgr=_tup(data.get("my_bgr")),
            other_bgr=_tup(data.get("other_bgr")),
        )


def _color_distance(a: tuple[int, int, int] | np.ndarray, b: tuple[int, int, int] | np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def bubble_extent(
    img: np.ndarray, box: Box, *, threshold: float = 10.0, max_scan: int = 900
) -> tuple[int, int] | None:
    """从文本框向外扫描，找出气泡的左右边界（同一行上颜色突变处）。

    这是「用几何而不是颜色」判断消息属于哪一侧的关键：长消息文本会换行，
    文本框右边界并不贴气泡边界，所以不能用文本框本身算留白。
    """
    h, w = img.shape[:2]
    y = int(min(max(box.cy, 0), h - 1))
    row = img[y].astype(float)

    start_left = min(max(box.x1 - 6, 0), w - 1)
    start_right = min(max(box.x2 + 6, 0), w - 1)

    def _scan(start: int, step: int) -> int | None:
        anchor = row[start]
        for i in range(1, max_scan):
            x = start + step * i
            if x < 0 or x >= w:
                return None
            if _color_distance(row[x], anchor) > threshold:
                return x - step
        return None

    left = _scan(start_left, -1)
    right = _scan(start_right, +1)
    if left is None or right is None or right <= left:
        return None
    return left, right


def _mode_of(samples: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    """样本里出现最多的颜色（至少 3 个样本）。"""
    clusters = _cluster_colors(samples)
    if not clusters:
        return None
    color, count = clusters[0]
    return color if count >= 3 else None


def _line_side(img: np.ndarray, box: Box, panel_left: int, panel_right: int) -> str | None:
    """用气泡边界与面板边界的留白判断消息属于哪一侧；无法判断返回 None。"""
    extent = bubble_extent(img, box)
    if extent is None:
        return None
    left, right = extent
    left_gap = left - panel_left
    right_gap = panel_right - right
    if left_gap < 0 or right_gap < 0:
        return None
    span = float(panel_right - panel_left)
    if min(left_gap, right_gap) > span * 0.35:  # 两侧留白都很大：大概率不是气泡
        return None
    if right_gap < left_gap * 0.7:
        return "me"
    if left_gap < right_gap * 0.7:
        return "other"
    return None


def _collect_side_colors(
    img: np.ndarray,
    lines: list[OcrLine],
    *,
    panel_left: int,
    panel_right: int,
    background: tuple[int, int, int],
    min_background_distance: float = 6.0,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """按侧别收集气泡颜色样本（过滤掉与背景几乎相同的采样块）。

    另外按侧别做物理约束：
    - 「我」的气泡一定是彩色的 → 丢弃低彩度采样（文件卡片等）；
    - 「对方」的气泡一定比聊天背景亮（白气泡/深色主题下略亮的灰）→ 丢弃更暗的卡片底色。
    """
    me: list[tuple[int, int, int]] = []
    other: list[tuple[int, int, int]] = []
    for line in lines:
        side = _line_side(img, line.box, panel_left, panel_right)
        if side is None:
            continue
        for patch, _weight in _sample_patches(line.box, min_px=3):
            mean = _patch_mean(img, patch)
            if mean is None:
                continue
            color = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))
            if _color_distance(color, background) < min_background_distance:
                continue
            if side == "me":
                if _chroma(color) < 10:  # 我的气泡一定是彩色
                    continue
                me.append(color)
            else:
                if _brightness(color) <= _brightness(background) + 0.5:  # 对方气泡比背景亮
                    continue
                other.append(color)
    return me, other


def _chroma(color: tuple[int, int, int]) -> int:
    """颜色鲜艳度：微信里「我」的气泡永远是彩色，「对方」永远接近白/灰。"""
    return max(color) - min(color)


def _cluster_colors(
    samples: list[tuple[int, int, int]], *, bin_size: int = 8, top_n: int = 12
) -> list[tuple[tuple[int, int, int], int]]:
    """把采样颜色按量化分箱聚类，返回 ``[(均值色, 样本数), ...]``（按样本数降序）。"""
    from collections import Counter

    if not samples:
        return []
    keys = [tuple(round(c / bin_size) * bin_size for c in s) for s in samples]
    counter = Counter(keys)
    clusters: list[tuple[tuple[int, int, int], int]] = []
    for key, count in counter.most_common(top_n):
        near = [s for s in samples if all(abs(s[i] - key[i]) <= bin_size * 2 for i in range(3))]
        mean = np.mean(np.array(near, dtype=float), axis=0)
        color = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))
        clusters.append((color, count))
    return clusters


def _brightness(color: tuple[int, int, int]) -> float:
    return float(sum(color)) / 3.0


def _pick_from_clusters(
    clusters: list[tuple[tuple[int, int, int], int]],
    background: tuple[int, int, int],
    *,
    min_votes: int = 3,
    min_chroma: int = 12,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]:
    """从颜色聚类里挑出「我」和「对方」的气泡色。

    - 我：出现最多的**彩色**聚类（微信里我的气泡是绿色/彩色）；
    - 对方：出现最多的**比背景更亮的近中性色**聚类。微信里对方气泡是白（浅色主题）
      或比背景略亮的灰（深色主题），而文件卡片/引用块通常比背景更暗，借此区分。
    """
    chromatic = [(c, n) for c, n in clusters if _chroma(c) >= min_chroma and n >= min_votes]
    achromatic = [
        (c, n)
        for c, n in clusters
        if _chroma(c) < min_chroma
        and n >= min_votes
        and _brightness(c) > _brightness(background) + 0.5
    ]
    my = max(chromatic, key=lambda item: item[1])[0] if chromatic else None
    other = max(achromatic, key=lambda item: item[1])[0] if achromatic else None
    return my, other


def _collect_all_colors(
    img: np.ndarray,
    lines: list[OcrLine],
    *,
    background: tuple[int, int, int],
    min_background_distance: float = 4.0,
) -> list[tuple[int, int, int]]:
    """收集所有采样块颜色（不区分左右），给颜色聚类兜底用。"""
    colors: list[tuple[int, int, int]] = []
    for line in lines:
        for patch, _weight in _sample_patches(line.box, min_px=3):
            mean = _patch_mean(img, patch)
            if mean is None:
                continue
            color = (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))
            if _color_distance(color, background) < min_background_distance:
                continue
            colors.append(color)
    return colors


def learn_palette(
    img: np.ndarray,
    message_rect: Box,
    lines: list[OcrLine],
    *,
    panel_left: int,
    panel_right: int,
) -> BubblePalette | None:
    """从一帧截图 + OCR 行里学习当前主题的气泡配色。"""
    bg = estimate_background_rgb(img, message_rect)
    if bg is None:
        return None
    background = (int(bg[0]), int(bg[1]), int(bg[2]))

    # 1) 几何自举：只采「左右留白差异明显」的短消息，颜色最干净
    me_samples, other_samples = _collect_side_colors(
        img, lines, panel_left=panel_left, panel_right=panel_right, background=background
    )
    geo_my = _mode_of(me_samples)
    geo_other = _mode_of(other_samples)

    # 2) 颜色聚类兜底：长消息两侧留白对称时几何法失效，改用「彩色=我 / 中性色=对方」
    all_colors = _collect_all_colors(img, lines, background=background)
    clusters = _cluster_colors(all_colors)
    cluster_my, cluster_other = _pick_from_clusters(clusters, background)

    return BubblePalette(
        background_bgr=background,
        my_bgr=geo_my or cluster_my,
        other_bgr=geo_other or cluster_other,
    )


class PaletteStore:
    """按主题指纹缓存配色，避免每次启动都要重新学习。"""

    def __init__(self, path: Path, *, max_entries: int = 8) -> None:
        self.path = Path(path)
        self.max_entries = max_entries

    def load(self) -> list[BubblePalette]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        palettes = []
        for item in raw.get("palettes", []):
            try:
                palettes.append(BubblePalette.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
        return palettes

    def save(self, palettes: list[BubblePalette]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"palettes": [p.to_dict() for p in palettes[-self.max_entries :]]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def match(self, background_bgr: np.ndarray | tuple[int, int, int], *, tolerance: float = 12.0):
        target = (
            (int(background_bgr[0]), int(background_bgr[1]), int(background_bgr[2]))
            if not isinstance(background_bgr, np.ndarray)
            else (int(background_bgr[0]), int(background_bgr[1]), int(background_bgr[2]))
        )
        best: tuple[float, BubblePalette] | None = None
        for palette in self.load():
            dist = _color_distance(palette.background_bgr, target)
            if dist <= tolerance and (best is None or dist < best[0]):
                best = (dist, palette)
        return None if best is None else best[1]

    def upsert(self, palette: BubblePalette) -> None:
        existing = [p for p in self.load() if p.theme_key != palette.theme_key]
        existing.append(palette)
        self.save(existing)
