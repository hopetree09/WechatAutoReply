"""窗口内区域划分：聊天面板 / 标题栏 / 消息区 / 输入区。

旧实现把「消息区 = 窗口高度 12%~72%」写死在配置里，窗口大小或布局变化就错位。
这里统一用「窗口比例 → 像素矩形」，并且把左侧会话列表排除在外（微信 4.x 的
聊天面板只占右侧一部分）。
"""

from __future__ import annotations

from wagent.config import LayoutSettings
from wagent.perception.types import Box


def chat_panel(img_w: int, img_h: int, layout: LayoutSettings) -> Box:
    """右侧聊天面板矩形。"""
    x1 = int(img_w * layout.chat_panel_left)
    return Box(x1, 0, img_w, img_h)


def title_area(img_w: int, img_h: int, layout: LayoutSettings) -> Box:
    """聊天面板顶部标题区（用于 OCR 会话名）。"""
    panel = chat_panel(img_w, img_h, layout)
    return Box(panel.x1, int(img_h * layout.title_top), panel.x2, int(img_h * layout.message_area_top))


def message_area(img_w: int, img_h: int, layout: LayoutSettings) -> Box:
    """消息区：标题栏之下、输入框之上。"""
    panel = chat_panel(img_w, img_h, layout)
    return Box(
        panel.x1,
        int(img_h * layout.message_area_top),
        panel.x2,
        int(img_h * layout.message_area_bottom),
    )


def input_area(img_w: int, img_h: int, layout: LayoutSettings) -> Box:
    """输入区（含底部工具栏）的搜索范围，供视觉定位找输入框。"""
    panel = chat_panel(img_w, img_h, layout)
    return Box(panel.x1, int(img_h * layout.message_area_bottom), panel.x2, img_h)
