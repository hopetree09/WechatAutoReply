"""LLM 评委 + A/B 对比（base 模型 vs LoRA 模型）。

评委只做「更像真人吗」的成对比较，不暴露哪个是微调模型，并支持交换 A/B 位置
来抵消位置偏差。解析失败/评委异常都降级为 ``tie``，不中断评测。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from wagent.eval.metrics import ReplyScore, aggregate, score_reply
from wagent.llm.client import ChatModel, MessageTurn

JUDGE_SYSTEM = """你是微信聊天风格评委。给你一段聊天记录和两条候选回复 A/B，
判断哪条更像「本人」随手打出来的回复：更口语、更短、更自然，没有 AI 味。

只输出 JSON：{"winner":"A"|"B"|"tie","reason":"一句话理由"}"""


@dataclass(frozen=True, slots=True)
class JudgeResult:
    winner: str
    reason: str = ""


def parse_judge(raw: str) -> JudgeResult:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            winner = str(data.get("winner", "")).strip().upper()
            if winner in {"A", "B", "TIE"}:
                return JudgeResult("tie" if winner == "TIE" else winner, str(data.get("reason", "")))
        except json.JSONDecodeError:
            pass
    # 兜底：从自由文本里找 A/B
    match = re.search(r"winner\s*[:：]\s*[\"']?([ABab]|tie)", text)
    if match:
        value = match.group(1).upper()
        return JudgeResult("tie" if value == "TIE" else value, text[:80])
    return JudgeResult("tie", text[:80])


@dataclass
class LLMJudge:
    """成对比较：内部交换 A/B 各评一次，两次一致才算赢。"""

    llm: ChatModel
    swap_check: bool = True

    def compare(self, context_text: str, reply_a: str, reply_b: str) -> JudgeResult:
        first = self._ask(context_text, reply_a, reply_b)
        if not self.swap_check:
            return first
        second = self._ask(context_text, reply_b, reply_a)
        # 交换后仍选同一个实际回复才算赢，否则视为平局（消除位置偏差）
        second_fixed = {"A": "B", "B": "A", "tie": "tie"}[second.winner]
        if first.winner != second_fixed:
            return JudgeResult("tie", f"位置偏差抵消：{first.winner}/{second_fixed}")
        return first

    def _ask(self, context_text: str, a: str, b: str) -> JudgeResult:
        user = (
            f"聊天记录：\n{context_text}\n\n"
            f"候选 A：{a}\n候选 B：{b}\n\n"
            "哪条更像本人？输出 JSON。"
        )
        try:
            result = self.llm.chat(
                [MessageTurn("system", JUDGE_SYSTEM), MessageTurn("user", user)],
                temperature=0.0,
                max_tokens=150,
            )
        except Exception:
            return JudgeResult("tie", "评委调用失败")
        return parse_judge(result.text)


@dataclass(frozen=True, slots=True)
class ABPrompt:
    """一次对比：同一段上下文 + 对方最新消息。"""

    context_text: str
    latest: str
    reference: str = ""  # 我当时的真实回复（用于风格相似度）


@dataclass(slots=True)
class ABResult:
    n: int = 0
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    base_metrics: dict[str, float] = field(default_factory=dict)
    tuned_metrics: dict[str, float] = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)

    def win_rate(self, side: str) -> float:
        decided = self.a_wins + self.b_wins
        if decided == 0:
            return 0.0
        return round((self.b_wins if side == "b" else self.a_wins) / decided, 3)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "b_win_rate": self.win_rate("b"),
            "base_metrics": self.base_metrics,
            "tuned_metrics": self.tuned_metrics,
            "details": self.details,
        }


Generate = Callable[[ABPrompt], str]


def run_ab(
    prompts: Sequence[ABPrompt],
    generate_base: Generate,
    generate_tuned: Generate,
    *,
    judge: LLMJudge | None = None,
    references: Sequence[str] = (),
    limit: int | None = None,
) -> ABResult:
    """跑 A/B：base 作 A、LoRA 作 B；同时算规则指标与（可选）评委胜负。"""
    result = ABResult()
    base_scores: list[ReplyScore] = []
    tuned_scores: list[ReplyScore] = []
    selected = list(prompts)[:limit] if limit else list(prompts)

    for prompt in selected:
        context_text = prompt.context_text
        reply_a = generate_base(prompt)
        reply_b = generate_tuned(prompt)
        refs = list(references) + ([prompt.reference] if prompt.reference else [])
        score_a = score_reply(reply_a, refs, context_texts=[context_text, prompt.latest])
        score_b = score_reply(reply_b, refs, context_texts=[context_text, prompt.latest])
        base_scores.append(score_a)
        tuned_scores.append(score_b)
        result.n += 1

        winner = "tie"
        reason = ""
        if judge is not None:
            verdict = judge.compare(
                f"{context_text}\n对方刚发来：{prompt.latest}", reply_a, reply_b
            )
            winner = verdict.winner
            reason = verdict.reason
        if winner == "A":
            result.a_wins += 1
        elif winner == "B":
            result.b_wins += 1
        else:
            result.ties += 1

        result.details.append(
            {
                "latest": prompt.latest,
                "context_text": context_text,
                "reference": prompt.reference,
                "base": score_a.as_dict(),
                "tuned": score_b.as_dict(),
                "winner": winner,
                "reason": reason,
            }
        )

    result.base_metrics = aggregate(base_scores)
    result.tuned_metrics = aggregate(tuned_scores)
    return result
