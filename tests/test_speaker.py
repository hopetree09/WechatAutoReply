"""说话人判别单测：合成气泡图，验证三分类投票与 UNKNOWN 语义。"""

from __future__ import annotations

import numpy as np

from wagent.config import BubbleSettings
from wagent.perception.speaker import (
    classify_speaker,
    estimate_background_rgb,
)
from wagent.perception.types import Box, Speaker

BG_LIGHT = (240, 240, 240)  # 浅色聊天区背景（RGB）
ME_GREEN = (149, 236, 105)
OTHER_WHITE = (255, 255, 255)

SETTINGS = BubbleSettings()


def make_image(
    bubble_rgb: tuple[int, int, int] | None,
    rect: tuple[int, int, int, int] | None,
    *,
    bg: tuple[int, int, int] = BG_LIGHT,
    size: tuple[int, int] = (600, 800),
) -> np.ndarray:
    """生成 BGR 合成图：背景 + 一块气泡。"""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bg[::-1]
    if bubble_rgb is not None and rect is not None:
        x1, y1, x2, y2 = rect
        img[y1:y2, x1:x2] = bubble_rgb[::-1]
    return img


def test_estimate_background() -> None:
    img = make_image(ME_GREEN, (460, 100, 760, 160))
    bg = estimate_background_rgb(img)
    assert bg is not None
    # 量化到 8 的倍数 +4，允许 ±4 误差
    assert np.allclose(bg, np.array(BG_LIGHT[::-1]), atol=4)


def test_me_bubble_detected() -> None:
    img = make_image(ME_GREEN, (460, 100, 760, 160))
    box = Box(500, 115, 740, 145)  # 文本在气泡内部，四周都有绿色内边距
    speaker, conf = classify_speaker(img, box, SETTINGS)
    assert speaker is Speaker.ME
    assert conf >= 0.5


def test_other_bubble_detected() -> None:
    img = make_image(OTHER_WHITE, (80, 100, 420, 160))
    box = Box(120, 115, 400, 145)
    speaker, conf = classify_speaker(img, box, SETTINGS)
    assert speaker is Speaker.OTHER
    assert conf >= 0.5


def test_chat_background_gives_unknown() -> None:
    """文字直接画在聊天背景上（系统提示/链接卡片）时，不能判成任何一方。"""
    img = make_image(None, None)
    speaker, conf = classify_speaker(img, Box(120, 115, 400, 145), SETTINGS)
    assert speaker is Speaker.UNKNOWN
    assert conf == 0.0


def test_dark_theme_gives_unknown() -> None:
    """未标定的深色主题（气泡色不在配置里）应判 UNKNOWN，而不是乱猜。"""
    dark_bg = (30, 30, 30)
    img = make_image((52, 52, 52), (80, 100, 420, 160), bg=dark_bg)
    speaker, _ = classify_speaker(img, Box(120, 115, 400, 145), SETTINGS)
    assert speaker is Speaker.UNKNOWN


def test_box_outside_image() -> None:
    img = make_image(None, None)
    speaker, _ = classify_speaker(img, Box(2000, 2000, 2100, 2050), SETTINGS)
    assert speaker is Speaker.UNKNOWN


def test_tight_padding_uses_vertical_padding() -> None:
    """气泡左右内边距很窄时，靠上下内边距采样仍能正确投票。"""
    img = make_image(ME_GREEN, (48, 0, 400, 400))
    box = Box(54, 115, 380, 145)
    speaker, _ = classify_speaker(img, box, SETTINGS)
    assert speaker is Speaker.ME


def test_explicit_background_overrides_estimate() -> None:
    img = make_image(OTHER_WHITE, (80, 100, 420, 160))
    box = Box(120, 115, 400, 145)
    speaker, _ = classify_speaker(img, box, SETTINGS, background_rgb=(240, 240, 240))
    assert speaker is Speaker.OTHER
