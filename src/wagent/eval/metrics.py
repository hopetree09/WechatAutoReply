"""回复质量/风格指标（不依赖模型，纯函数，便于单测与回归）。

评测分三层：
1. **风格相似度**：生成回复与「我真实说过的话」的字级 2-gram Jaccard；
2. **AI 味检测**：模板化连接词、说教腔、过长句号、过度礼貌，越少越好；
3. **长度自然度**：聊天回复通常 3~30 字，过长扣分。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

AI_FLAVOR_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"首先",
        r"其次",
        r"最后",
        r"总之",
        r"总的来说",
        r"总结",
        r"综上所述",
        r"需要注意的是",
        r"值得注意",
        r"希望(?:这|以上|对你)",
        r"作为(?:一个)?\s*AI",
        r"很高兴(?:为你|为能)",
        r"我可以(?:帮|为)你",
        r"如果你(?:有|需要).{0,6}(?:随时|可以)",
        r"建议你",
        r"亲[，,]",
        r"您",
        r"加油哦",
    )
)

POLITE_PATTERNS = (re.compile(r"^您好"), re.compile(r"请问"), re.compile(r"非常感谢"))


def char_ngrams(text: str, n: int = 2) -> set[str]:
    """中文字级 n-gram（忽略标点与空白）。"""
    cleaned = re.sub(r"[\s\W_]+", "", text)
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard(a: str, b: str, *, n: int = 2) -> float:
    grams_a, grams_b = char_ngrams(a, n), char_ngrams(b, n)
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


def style_similarity(reply: str, references: Sequence[str], *, top_k: int = 3) -> float:
    """与「我真实说过的话」的相似度。

    取最像的 top_k 句，但只对「有实际重叠（>=0.15）」的句子求平均：
    否则参考句一多，平均值会被大量不相关句子稀释，指标失去区分度。
    """
    if not reply or not references:
        return 0.0
    scores = sorted((jaccard(reply, ref) for ref in references), reverse=True)[:top_k]
    strong = [score for score in scores if score >= 0.15]
    if strong:
        return round(sum(strong) / len(strong), 3)
    return round(scores[0] if scores else 0.0, 3)


def ai_flavor_score(text: str) -> float:
    """AI 味得分：0 最好，1 最差。"""
    if not text:
        return 1.0
    hits = sum(1 for pattern in AI_FLAVOR_PATTERNS if pattern.search(text))
    hits += sum(1 for pattern in POLITE_PATTERNS if pattern.search(text))
    # 句号/分号越多越像正式书面语
    formal = len(re.findall(r"[。；;]", text))
    length = max(1, len(text))
    score = hits * 0.25 + min(1.0, formal * 0.3)
    # 长句 + 连接词密集，进一步加权
    if hits and length > 40:
        score = min(1.0, score + 0.2)
    return round(min(1.0, score), 3)


def length_naturalness(reply: str, *, soft_min: int = 2, soft_max: int = 30) -> float:
    """聊天回复长度自然度：区间内 1.0，越长越扣分。"""
    length = len(reply.strip())
    if length == 0:
        return 0.0
    if soft_min <= length <= soft_max:
        return 1.0
    if length < soft_min:
        return 0.7
    return max(0.0, 1.0 - (length - soft_max) / 60.0)


def repeat_penalty(reply: str, context_texts: Sequence[str]) -> float:
    """和对方原话/上文高度重复扣分（复读机检测）。"""
    for text in context_texts:
        if text and (text.strip() == reply.strip() or jaccard(reply, text, n=3) > 0.85):
            return 0.5
    return 1.0


@dataclass(slots=True)
class ReplyScore:
    reply: str
    style: float = 0.0
    ai_flavor: float = 0.0
    length: float = 0.0
    repetition: float = 1.0
    score: float = 0.0
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "reply": self.reply,
            "style": round(self.style, 3),
            "ai_flavor": round(self.ai_flavor, 3),
            "length": round(self.length, 3),
            "repetition": round(self.repetition, 3),
            "score": round(self.score, 3),
            **{k: round(v, 3) for k, v in self.details.items()},
        }


def score_reply(
    reply: str,
    references: Sequence[str] = (),
    context_texts: Sequence[str] = (),
) -> ReplyScore:
    """综合打分：风格 50% + 自然度 25% + 反 AI 味 25%，再乘复读惩罚。"""
    style = style_similarity(reply, references)
    flavor = ai_flavor_score(reply)
    natural = length_naturalness(reply)
    repeat = repeat_penalty(reply, context_texts)
    base = 0.5 * style + 0.25 * natural + 0.25 * (1.0 - flavor)
    return ReplyScore(
        reply=reply,
        style=style,
        ai_flavor=flavor,
        length=natural,
        repetition=repeat,
        score=round(base * repeat, 3),
    )


def aggregate(scores: Sequence[ReplyScore]) -> dict[str, float]:
    """汇总一组打分。"""
    if not scores:
        return {"n": 0, "score": 0.0, "style": 0.0, "ai_flavor": 0.0, "length": 0.0}
    n = len(scores)
    return {
        "n": n,
        "score": round(sum(s.score for s in scores) / n, 3),
        "style": round(sum(s.style for s in scores) / n, 3),
        "ai_flavor": round(sum(s.ai_flavor for s in scores) / n, 3),
        "length": round(sum(s.length for s in scores) / n, 3),
    }
