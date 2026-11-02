"""意图路由：先判断「这条消息该不该回、该怎么回」。

为什么要单独做：如果每条消息都回，遇到「嗯」「表情包」会显得很傻，遇到转账/验证码更是安全事故。
所以主循环在触发生成前先做一次轻量意图判定：

    NO_REPLY  → 不回（语气词、表情、纯符号、红包提示等）
    SENSITIVE → 不回（金钱/验证码/隐私相关，交给人）
    PLAN      → 约时间/地点，回复要落到具体安排
    QUESTION  → 问题，正常回答
    CHAT      → 闲聊

默认用规则（零成本、可解释）；可选叠加 LLM 分类器处理规则拿不准的情况。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from wagent.llm.client import ChatModel as ChatModelLike


class Intent(StrEnum):
    NO_REPLY = "no_reply"
    SENSITIVE = "sensitive"
    PLAN = "plan"
    QUESTION = "question"
    CHAT = "chat"


@dataclass(frozen=True, slots=True)
class IntentResult:
    intent: Intent
    confidence: float = 0.6
    reason: str = ""


# 明显不需要回的消息（语气词、表情、纯符号、系统提示）
_NO_REPLY_TEXTS = frozenset(
    {
        "嗯", "嗯嗯", "哦", "哦哦", "好", "好的", "好滴", "行", "ok", "OK", "收到",
        "哈哈", "哈哈哈", "哈哈哈哈哈", "笑死", "表情", "表情包", "赞", "👍", "😂",
    }
)

_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"转账", r"红包", r"收款", r"付款", r"借钱", r"借我", r"还钱", r"扫码付",
        r"验证码", r"密码", r"身份证", r"银行卡", r"卡号", r"实名", r"登录码",
        r"帮我付", r"代付", r"垫付",
    )
)

_PLAN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"几点", r"什么时候", r"哪天", r"周末", r"明天", r"后天", r"今晚", r"下午",
        r"见面", r"约", r"有空吗", r"来不来", r"一起", r"老地方", r"定个时间",
    )
)

_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (r"[?？]\s*$", r"吗\s*$", r"怎么", r"为什么", r"多少", r"哪个", r"哪里", r"如何", r"能不能", r"是不是")
)

_EMOJI_ONLY = re.compile(r"^[\U0001F300-\U0001FAFF\U00002600-\U000027BF\s，。,.!！?？~～]+$")


def classify_rules(text: str) -> IntentResult:
    """零成本规则分类。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return IntentResult(Intent.NO_REPLY, 1.0, "空消息")
    if cleaned in _NO_REPLY_TEXTS or len(cleaned) <= 1:
        return IntentResult(Intent.NO_REPLY, 0.95, "语气词/单字")
    if _EMOJI_ONLY.match(cleaned):
        return IntentResult(Intent.NO_REPLY, 0.9, "纯表情/符号")
    if any(p.search(cleaned) for p in _SENSITIVE_PATTERNS):
        return IntentResult(Intent.SENSITIVE, 0.9, "金钱/隐私相关")
    if any(p.search(cleaned) for p in _PLAN_PATTERNS):
        return IntentResult(Intent.PLAN, 0.75, "约时间/地点")
    if any(p.search(cleaned) for p in _QUESTION_PATTERNS):
        return IntentResult(Intent.QUESTION, 0.7, "提问")
    return IntentResult(Intent.CHAT, 0.5, "闲聊")


_INTENT_SYSTEM = """你是意图分类器。判断对方这条微信消息属于哪类：
- no_reply：只是语气词/表情/纯符号，不需要回复
- sensitive：涉及转账、红包、借钱、验证码、密码、身份证、银行卡等金钱或隐私
- plan：约时间、地点、见面
- question：提问、求助
- chat：普通闲聊
只输出 JSON：{"intent":"...","confidence":0.0~1.0}"""


def parse_intent_json(raw: str) -> IntentResult | None:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    value = str(data.get("intent", "")).strip().lower()
    if value not in {intent.value for intent in Intent}:
        return None
    try:
        confidence = float(data.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    return IntentResult(Intent(value), max(0.0, min(1.0, confidence)), "llm")


# 记忆注入的触发词：问到"之前/上次/配置/几点/哪/多少"这类才值得翻记忆
RECALL_HINTS = (
    "记得",
    "之前",
    "上次",
    "以前",
    "说过",
    "聊过",
    "配置",
    "显卡",
    "几点",
    "哪天",
    "哪里",
    "哪儿",
    "多少",
    "什么",
    "怎么样",
    "咋样",
    "约好",
    "安排",
    "地址",
    "生日",
    "名字",
)


def needs_memory(text: str, intent: IntentResult | None = None) -> bool:
    """按意图决定要不要注入层级记忆（避免闲聊里塞无关事实答非所问）。

    - 提问 / 约时间：注入（可能涉及历史信息）；
    - 语气词 / 敏感话题：不注入；
    - 其他闲聊：只有话里出现"回忆类"关键词才注入。
    """
    result = intent or classify_rules(text)
    if result.intent in (Intent.NO_REPLY, Intent.SENSITIVE):
        return False
    if result.intent in (Intent.QUESTION, Intent.PLAN):
        return True
    return any(hint in text for hint in RECALL_HINTS)


@dataclass
class IntentRouter:
    """规则优先、LLM 兜底的意图路由。"""

    llm: ChatModelLike | None = None
    use_llm: bool = False
    llm_min_confidence: float = 0.55

    def classify(self, text: str) -> IntentResult:
        rule = classify_rules(text)
        if not self.use_llm or self.llm is None:
            return rule
        # 规则已经很有把握就不花 LLM 的钱
        if rule.confidence >= 0.9:
            return rule
        try:
            result = self.llm.chat(
                [
                    {"role": "system", "content": _INTENT_SYSTEM},
                    {"role": "user", "content": f"消息：{text}"},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            parsed = parse_intent_json(getattr(result, "text", ""))
        except Exception:
            return rule
        if parsed is None or parsed.confidence < self.llm_min_confidence:
            return rule
        return parsed
