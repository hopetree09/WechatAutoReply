"""把屏幕上的时间戳文本解析成绝对时间。

微信聊天里时间戳有好几种写法：``21:03``、``昨天 21:03``、``10月20日 08:21``、
``星期五 15:15``、``2025年10月20日 21:03``。采史/入库时需要把它们换算成绝对时间。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

_CLOCK = r"(\d{1,2}):(\d{2})"

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"^{_CLOCK}$"), "clock"),
    (re.compile(rf"^(今天|昨天|前天)\s*{_CLOCK}$"), "day_only"),
    (re.compile(rf"^(上午|下午|中午|晚上|凌晨|早上)\s*{_CLOCK}$"), "clock_ampm"),
    (re.compile(r"^(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?(?:\s*(\d{1,2}):(\d{2}))?$"), "ymd"),
    (re.compile(r"^(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?$"), "md"),
    (re.compile(r"^(?:星期|周)([一二三四五六日天])(?:\s*(\d{1,2}):(\d{2}))?$"), "weekday"),
    (re.compile(r"^(\d{1,2})[/-](\d{1,2})(?:\s*(\d{1,2}):(\d{2}))?$"), "md"),
)


def _at(base: datetime, hour: int, minute: int) -> datetime:
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def resolve_ts_text(text: str, now: datetime) -> str | None:
    """把屏幕时间戳解析成 ISO 字符串；无法识别返回 ``None``。"""
    raw = (text or "").strip().replace(" ", "")
    if not raw:
        return None
    # 归一化：全角数字/冒号
    raw = raw.translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))

    for pattern, kind in _PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        groups = match.groups()
        try:
            if kind == "clock":
                return _at(now, int(groups[0]), int(groups[1])).isoformat(timespec="seconds")
            if kind == "day_only":
                offset = {"今天": 0, "昨天": 1, "前天": 2}[groups[0]]
                target_day = now - timedelta(days=offset)
                return _at(target_day, int(groups[1]), int(groups[2])).isoformat(timespec="seconds")
            if kind == "clock_ampm":
                hour, minute = int(groups[1]), int(groups[2])
                period = groups[0]
                if period in {"下午", "晚上"} and hour < 12:
                    hour += 12
                if period == "凌晨" and hour == 12:
                    hour = 0
                return _at(now, hour, minute).isoformat(timespec="seconds")
            if kind == "ymd":
                year, month, day_num = int(groups[0]), int(groups[1]), int(groups[2])
                hour = int(groups[3]) if groups[3] else 0
                minute = int(groups[4]) if groups[4] else 0
                return datetime(year, month, day_num, hour, minute).astimezone().isoformat(
                    timespec="seconds"
                )
            if kind == "md":
                month, day_num = int(groups[0]), int(groups[1])
                hour = int(groups[2]) if groups[2] else 0
                minute = int(groups[3]) if groups[3] else 0
                # 聊天记录里的日期永远是过去：算出来比 now 晚就归到去年
                candidate = datetime(now.year, month, day_num, hour, minute).astimezone()
                if candidate > now:
                    candidate = candidate.replace(year=now.year - 1)
                return candidate.isoformat(timespec="seconds")
            if kind == "weekday":
                target = _WEEKDAYS[groups[0]]
                hour = int(groups[1]) if groups[1] else 0
                minute = int(groups[2]) if groups[2] else 0
                delta = (now.weekday() - target) % 7
                target_day = now - timedelta(days=delta)
                return _at(target_day, hour, minute).isoformat(timespec="seconds")
        except (ValueError, KeyError):
            return None
    return None
