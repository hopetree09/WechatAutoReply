"""气泡配色自举学习单测。"""

from __future__ import annotations

import numpy as np

from wagent.perception.calibration import (
    BubblePalette,
    PaletteStore,
    bubble_extent,
    learn_palette,
)
from wagent.perception.types import Box, OcrLine

BG = (240, 240, 240)
ME_GREEN = (157, 242, 159)
OTHER_WHITE = (255, 255, 255)

PANEL_LEFT = 0
PANEL_RIGHT = 800


def make_chat() -> np.ndarray:
    """构造一帧聊天图：右侧绿色气泡（我）+ 左侧白色气泡（对方）。"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:, :] = BG[::-1]
    img[100:240, 420:700] = ME_GREEN[::-1]  # 我的长气泡（多行）
    img[320:400, 60:340] = OTHER_WHITE[::-1]  # 对方气泡
    return img


def make_lines() -> list[OcrLine]:
    return [
        OcrLine("我的第一行", 0.98, Box(440, 120, 680, 150)),
        OcrLine("我的第二行", 0.98, Box(440, 160, 560, 190)),
        OcrLine("我的第三行", 0.98, Box(440, 200, 500, 230)),
        OcrLine("对方说的", 0.98, Box(80, 340, 320, 370)),
    ]


def test_bubble_extent_finds_edges() -> None:
    img = make_chat()
    extent = bubble_extent(img, Box(440, 120, 680, 150))
    assert extent is not None
    left, right = extent
    assert abs(left - 420) <= 2
    assert abs(right - 700) <= 2


def test_learn_palette_names_sides_correctly() -> None:
    img = make_chat()
    message_rect = Box(0, 0, 800, 600)
    palette = learn_palette(
        img, message_rect, make_lines(), panel_left=PANEL_LEFT, panel_right=PANEL_RIGHT
    )
    assert palette is not None
    assert palette.my_bgr is not None and palette.other_bgr is not None
    # 学习到的颜色应与合成图一致（BGR）
    assert np.allclose(palette.my_bgr, ME_GREEN[::-1], atol=6)
    assert np.allclose(palette.other_bgr, OTHER_WHITE[::-1], atol=6)
    assert np.allclose(palette.background_bgr, BG[::-1], atol=6)


def test_learn_palette_skips_ambiguous_lines() -> None:
    img = make_chat()
    # 把一行放在正中间、且两侧留白接近 → 不参与学习
    lines = [OcrLine("居中", 0.9, Box(350, 500, 500, 530))]
    palette = learn_palette(
        img, Box(0, 0, 800, 600), lines, panel_left=PANEL_LEFT, panel_right=PANEL_RIGHT
    )
    assert palette is not None
    assert palette.my_bgr is None and palette.other_bgr is None


def test_gray_card_is_not_learned_as_other_bubble() -> None:
    """浅灰的文件卡片/引用块不能被学成「对方气泡色」，否则卡片文字会被当成聊天消息。"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:, :] = (250, 250, 250)[::-1]  # 近白背景
    img[100:240, 420:700] = ME_GREEN[::-1]  # 我的绿色气泡
    img[300:460, 80:420] = (238, 238, 238)[::-1]  # 浅灰卡片
    lines = [
        OcrLine("我的消息", 0.98, Box(440, 120, 680, 150)),
        OcrLine("卡片标题", 0.98, Box(100, 320, 300, 350)),
    ]
    palette = learn_palette(
        img, Box(0, 0, 800, 600), lines, panel_left=PANEL_LEFT, panel_right=PANEL_RIGHT
    )
    assert palette is not None
    assert palette.my_bgr is not None
    assert palette.other_bgr is None


def test_cluster_fallback_learns_my_color_without_side_info() -> None:
    """屏幕上只有自己的长消息（两侧留白对称）时，仍能靠「彩色=我」学到气泡色。"""
    img = make_chat()
    lines = [OcrLine("我的长消息第一行", 0.98, Box(440, 120, 680, 150))]
    palette = learn_palette(
        img, Box(0, 0, 800, 600), lines, panel_left=PANEL_LEFT, panel_right=PANEL_RIGHT
    )
    assert palette is not None
    assert palette.my_bgr is not None
    assert np.allclose(palette.my_bgr, ME_GREEN[::-1], atol=8)


def test_palette_store_roundtrip(tmp_path) -> None:
    store = PaletteStore(tmp_path / "calibration.json")
    palette = BubblePalette(background_bgr=(252, 252, 252), my_bgr=(159, 242, 157))
    store.upsert(palette)
    matched = store.match((252, 252, 252))
    assert matched is not None
    assert matched.my_bgr == (159, 242, 157)
    assert store.match((30, 30, 30)) is None


def test_palette_store_replaces_same_theme(tmp_path) -> None:
    store = PaletteStore(tmp_path / "calibration.json")
    store.upsert(BubblePalette(background_bgr=(252, 252, 252), my_bgr=(1, 2, 3)))
    store.upsert(BubblePalette(background_bgr=(252, 252, 252), my_bgr=(4, 5, 6)))
    assert len(store.load()) == 1
    assert store.match((252, 252, 252)).my_bgr == (4, 5, 6)  # type: ignore[union-attr]
