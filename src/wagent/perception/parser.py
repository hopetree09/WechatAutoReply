"""把 OCR 行解析成结构化消息。

这一步修的是旧实现的核心问题：
1. **不排序**：旧代码直接用 OCR 返回顺序，``msgs[-1]`` 未必是最新一条；
2. **不合并**：一条多行消息会被拆成多条 Message，导致「最后一行」判断错位；
3. **不过滤**：时间戳（``昨天 21:03``）、系统提示（``撤回了一条消息``）会被当成
   对方说的话，从而触发莫名其妙的回复。

解析流程：``过滤 → 按 y 排序 → 分组（同说话人且行距足够近）→ 合并文本``。
``classify`` 由调用方注入（基于气泡颜色的分类器），便于单测时用桩函数。
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

from wagent.perception.types import Box, Message, OcrLine, Speaker, row_bucket

# ---------------------------------------------------------------- 文本分类

TIMESTAMP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^\d{1,2}:\d{2}$",
        r"^(昨天|前天|今天|上午|下午|中午|晚上|凌晨|早上)\s*\d{1,2}:\d{2}$",
        r"^\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}日?(\s*\d{1,2}:\d{2})?$",
        r"^\d{1,2}月\d{1,2}日(\s*\d{1,2}:\d{2})?$",
        r"^(星期|周)[一二三四五六日天](\s*\d{1,2}:\d{2})?$",
        r"^\d{2,4}[/-]\d{1,2}[/-]\d{1,2}(\s+\d{1,2}:\d{2})?$",
        r"^\d{1,2}:\d{2}:\d{2}$",
    )
)

# 纯占位类文本（媒体消息、红包、转账等），默认不参与回复
MEDIA_PLACEHOLDERS = frozenset(
    {
        "图片",
        "文件",
        "语音",
        "视频",
        "动画表情",
        "表情",
        "链接",
        "位置",
        "名片",
        "音乐",
        "小程序",
        "转账",
        "红包",
        "聊天记录",
        "拍一拍",
        "[图片]",
        "[文件]",
        "[语音]",
        "[视频]",
        "[动画表情]",
        "[表情]",
        "[链接]",
        "[位置]",
        "[名片]",
        "[音乐]",
        "[小程序]",
        "[转账]",
        "[红包]",
        "[聊天记录]",
        "[微信红包]",
    }
)

# 出现即视为系统提示（不是聊天内容）
SYSTEM_KEYWORDS: tuple[str, ...] = (
    "撤回了一条消息",
    "以下为新消息",
    "对方正在输入",
    "拍了拍",
    "加入了群聊",
    "移出了群聊",
    "开启了朋友验证",
    "发送朋友验证",
    "你已添加了",
    "现在可以开始聊天",
    "被对方拒收",
    "该消息类型暂不支持",
    "本条消息已过期",
)

TextKind = Literal["message", "timestamp", "system", "noise"]


def normalize_text(text: str) -> str:
    """去掉零宽字符与多余空白；保留英文单词之间的单个空格。"""
    cleaned = text.replace("\u200b", "").replace("\ufeff", "").replace("\u3000", " ")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def classify_text(text: str) -> TextKind:
    """判断一行文本属于「聊天内容 / 时间戳 / 系统提示 / 噪声」。"""
    t = normalize_text(text)
    if not t:
        return "noise"
    if t in MEDIA_PLACEHOLDERS:
        return "system"
    if any(k in t for k in SYSTEM_KEYWORDS):
        return "system"
    if any(p.match(t) for p in TIMESTAMP_PATTERNS):
        return "timestamp"
    # 纯数字（未读红点、序号、日期碎片）
    if re.fullmatch(r"\d{1,4}", t):
        return "noise"
    # 纯符号（分隔线、装饰）
    if re.fullmatch(r"[\W_]+", t):
        return "noise"
    return "message"


# ---------------------------------------------------------------- 分组参数


@dataclass(frozen=True, slots=True)
class ParseConfig:
    min_score: float = 0.5
    # 行间垂直间隙 <= merge_gap_ratio * 行高 → 视为同一条消息的多行
    merge_gap_ratio: float = 0.6
    # 同一消息内各行的水平偏移容差（像素），避免把不同气泡并到一起
    x_tolerance: int = 12
    min_line_height: int = 8
    # 消息区左侧边界（相对窗口宽度的比例换算后的像素值），用于给「上下对齐」兜底
    content_left: int = 0
    content_right: int = 0


@dataclass(slots=True)
class _Group:
    speaker: Speaker
    lines: list[OcrLine] = field(default_factory=list)

    def push(self, line: OcrLine) -> None:
        self.lines.append(line)

    @property
    def last(self) -> OcrLine:
        return self.lines[-1]

    @property
    def first(self) -> OcrLine:
        return self.lines[0]


Classify = Callable[[Box], tuple[Speaker, float]]
Extent = Callable[[Box], tuple[int, int] | None]


def _gap(cur: OcrLine, nxt: OcrLine) -> float:
    """上一行的底边到下一行的顶边的垂直距离（允许负值=细微重叠）。"""
    return float(nxt.box.y1 - cur.box.y2)


def parse_ocr_lines(
    lines: Sequence[OcrLine],
    classify: Classify,
    *,
    session: str = "",
    cfg: ParseConfig | None = None,
    extent: Extent | None = None,
) -> list[Message]:
    """OCR 行 → 消息列表（按屏幕从上到下）。

    Args:
        lines: OCR 结果，可以无序。
        classify: ``Box -> (Speaker, confidence)``，由气泡分类器提供。
        session: 当前会话名，用于去重指纹。
        cfg: 解析参数。
        extent: 可选，``Box -> (气泡左边界, 右边界)``。提供后用它判断「两行是否
            属于同一个气泡」——同气泡内换行要合并，相邻独立气泡不能合并；仅靠行距
            无法区分这两种情况（聊天里相邻气泡间距可能比行内行距还小）。
    """
    cfg = cfg or ParseConfig()
    extent_cache: dict[int, tuple[int, int] | None] = {}
    group_extent: dict[int, tuple[int, int] | None] = {}

    # 1. 过滤噪声行
    valid: list[OcrLine] = []
    for ln in lines:
        text = normalize_text(ln.text)
        if ln.score < cfg.min_score or not text:
            continue
        if ln.box.h < cfg.min_line_height:
            continue
        valid.append(OcrLine(text=text, score=ln.score, box=ln.box))

    # 2. 按 y（再按 x）排序——旧实现缺这一步
    valid.sort(key=lambda ln: (ln.box.cy, ln.box.cx))

    # 3. 分组：遇到时间戳/系统提示就断开，防止跨时段误合并
    groups: list[_Group] = []
    current: _Group | None = None

    def _extent_of(ln: OcrLine) -> tuple[int, int] | None:
        if extent is None:
            return None
        key = id(ln)
        if key not in extent_cache:
            extent_cache[key] = extent(ln.box)
        return extent_cache[key]

    for ln in valid:
        kind = classify_text(ln.text)
        if kind != "message":
            current = None
            continue

        speaker, conf = classify(ln.box)
        if speaker is Speaker.UNKNOWN:
            current = None
            continue

        if current is not None and current.speaker is speaker:
            gap = _gap(current.last, ln)
            line_h = max(1.0, statistics.fmean(line.box.h for line in current.lines))
            same_column = abs(current.last.box.x1 - ln.box.x1) <= cfg.x_tolerance or (
                abs(current.last.box.x2 - ln.box.x2) <= cfg.x_tolerance
            )
            same_bubble = same_column
            gap_limit = cfg.merge_gap_ratio * line_h
            if extent is not None:
                cur_extent = group_extent[id(current)]
                new_extent = _extent_of(ln)
                same_bubble = (
                    cur_extent is not None
                    and new_extent is not None
                    and abs(cur_extent[0] - new_extent[0]) <= 6
                    and abs(cur_extent[1] - new_extent[1]) <= 6
                )
                if same_bubble:
                    # 同一气泡内换行：允许稍大的行距（多行文字的行距可能被 OCR 框放大）
                    gap_limit = max(gap_limit, 1.2 * line_h)
            if gap <= gap_limit and same_bubble:
                current.push(ln)
                continue
        current = _Group(speaker=speaker, lines=[ln])
        group_extent[id(current)] = _extent_of(ln)
        groups.append(current)

    # 4. 组内合并成消息
    messages: list[Message] = []
    for group in groups:
        # 中文换行只是排版，直接拼接；英文单词之间保留原空格
        text = ""
        for i, line in enumerate(group.lines):
            if i and text and text[-1].isascii() and line.text[0].isascii():
                text += " "
            text += line.text
        _, conf = classify(group.first.box)
        messages.append(
            Message(
                speaker=group.speaker,
                text=normalize_text(text),
                box=Box(
                    x1=min(line.box.x1 for line in group.lines),
                    y1=min(line.box.y1 for line in group.lines),
                    x2=max(line.box.x2 for line in group.lines),
                    y2=max(line.box.y2 for line in group.lines),
                ),
                confidence=round(conf, 3),
                session=session,
                row_bucket=row_bucket(group.first.box.cy),
            )
        )
    return messages


def pick_session_name(lines: Sequence[OcrLine], min_score: float = 0.5) -> str:
    """从聊天区顶部标题栏的 OCR 结果里挑会话名。

    简化策略：过滤时间戳/系统文本后，取「最靠上、字数在 1~30 之间」的一条。
    """
    candidates = [
        ln
        for ln in lines
        if ln.score >= min_score and 1 <= len(normalize_text(ln.text)) <= 30
        and classify_text(ln.text) == "message"
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda ln: (ln.box.cy, ln.box.cx))
    return normalize_text(candidates[0].text)
