"""安全护栏的实现：输出侧校验（发送前最后一道闸）。

三层：
1. **输入侧**：识别疑似提示词注入（对方让你"忽略规则/复述提示词"），只标记/告警，
   不改变聊天内容本身；prompt 里另有信任边界声明；
2. **输出侧**：金钱/隐私话题直接拦截；PII（手机号/身份证/邮箱/卡号）脱敏；
   复述对方原话、超长、空回复拦截；
3. **执行侧**：真正发送前再校验一次。

原则：宁可少发一条，也不发错一条。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"忽略(?:以上|上面|之前|前面)(?:的)?(?:所有)?(?:规则|指令|提示|要求)",
        r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions",
        r"重复(?:我)?(?:上面|以上|之前)的?(?:话|指令|提示词)",
        r"(?:输出|告诉我|打印)(?:你的)?(?:系统)?(?:提示词|prompt|指令)",
        r"你现在(?:是|扮演)",
        r"system\s*prompt",
    )
)

MONEY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"转账",
        r"红包",
        r"收款",
        r"付款",
        r"转(?:你|我|他|她)",
        r"借(?:我|你)?钱",
        r"还钱",
        r"打钱",
        r"代付",
        r"垫付",
        r"扫码付",
        r"给(?:你|他|她)\s*\d+\s*(?:块|元|万)",
        r"验证码",
        r"密码",
        r"银行卡",
        r"卡号",
    )
)

PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"1[3-9]\d{9}"), "***手机号***"),
    (re.compile(r"\d{17}[\dXx]"), "***身份证***"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "***邮箱***"),
    (re.compile(r"\b\d{16,19}\b"), "***卡号***"),
)

MAX_REPLY_CHARS = 220
_LAUGHTER = re.compile(r"^(?:哈|呵|嘿|笑)+[哈呵嘿笑。！!~～]*$")


@dataclass(frozen=True, slots=True)
class GuardResult:
    allowed: bool
    reason: str = ""
    safe_text: str = ""


def has_injection(text: str) -> bool:
    """输入侧：是否疑似提示词注入。"""
    return any(pattern.search(text or "") for pattern in INJECTION_PATTERNS)


def mask_pii(text: str) -> tuple[str, int]:
    """把 PII 替换为占位符，返回 ``(脱敏文本, 替换次数)``。"""
    masked = 0
    out = text
    for pattern, replacement in PII_PATTERNS:
        out, count = pattern.subn(replacement, out)
        masked += count
    return out, masked


def is_laughter(text: str) -> bool:
    return bool(_LAUGHTER.match((text or "").strip()))


def check_output(reply: str, *, context_texts: tuple[str, ...] = ()) -> GuardResult:
    """输出侧校验：返回是否允许发送 + 脱敏后的文本 + 原因。"""
    text = (reply or "").strip()
    if not text:
        return GuardResult(False, "空回复")
    if len(text) > MAX_REPLY_CHARS:
        return GuardResult(False, f"回复过长（{len(text)} 字）")
    if any(pattern.search(text) for pattern in MONEY_PATTERNS):
        return GuardResult(False, "涉及金钱/隐私话题，已拦截")
    if has_injection(text):
        return GuardResult(False, "回复疑似复述提示词/注入内容，已拦截")
    for context in context_texts:
        if not context:
            continue
        if text == context.strip() and not is_laughter(text):
            return GuardResult(False, "回复与对方原话完全重复，已拦截")
    safe_text, masked = mask_pii(text)
    if masked:
        return GuardResult(True, f"已脱敏 {masked} 处 PII", safe_text)
    return GuardResult(True, "", safe_text)
