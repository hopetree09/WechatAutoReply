"""评测层单测：指标、评委解析、A/B 汇总（全部纯逻辑，不需要模型）。"""

from __future__ import annotations

from wagent.eval.ab import ABPrompt, LLMJudge, parse_judge, run_ab
from wagent.eval.metrics import (
    ReplyScore,
    aggregate,
    ai_flavor_score,
    jaccard,
    length_naturalness,
    repeat_penalty,
    score_reply,
    style_similarity,
)

REFS = ["行吧 先这样", "问题不大 我看看", "哈哈可以"]


# ---------------------------------------------------------------- 指标


def test_jaccard_and_style_similarity() -> None:
    assert jaccard("行吧", "行吧") == 1.0
    assert jaccard("行吧", "你好呀") < 0.2
    assert style_similarity("行吧 先这样", REFS) > 0.5
    assert style_similarity("", REFS) == 0.0
    assert style_similarity("行吧", []) == 0.0


def test_ai_flavor_score() -> None:
    assert ai_flavor_score("首先，我们需要明确目标。其次，需要注意时间。") >= 0.5
    assert ai_flavor_score("行的 明早见") <= 0.1
    assert ai_flavor_score("") == 1.0
    assert ai_flavor_score("您好，请问有什么可以帮您？") >= 0.4


def test_length_naturalness() -> None:
    assert length_naturalness("好的") == 1.0
    assert length_naturalness("") == 0.0
    assert length_naturalness("一") == 0.7
    long_reply = "很长的回复" * 40
    assert length_naturalness(long_reply) < 0.6


def test_repeat_penalty() -> None:
    assert repeat_penalty("在吗", ["在吗"]) == 0.5
    assert repeat_penalty("在的", ["在吗"]) == 1.0


def test_score_reply_and_aggregate() -> None:
    good = score_reply("行吧 先这样", REFS)
    bad = score_reply("首先，我需要总结一下：这是一个很好的问题。", REFS)
    assert good.score > bad.score
    summary = aggregate([good, bad])
    assert summary["n"] == 2
    assert 0 < summary["score"] <= 1
    assert set(summary) == {"n", "score", "style", "ai_flavor", "length"}
    assert aggregate([])["n"] == 0
    assert isinstance(ReplyScore("x").as_dict(), dict)


# ---------------------------------------------------------------- 评委


def test_parse_judge_variants() -> None:
    assert parse_judge('{"winner":"A","reason":"更短"}').winner == "A"
    assert parse_judge('```json\n{"winner":"B"}\n```').winner == "B"
    assert parse_judge('winner: A').winner == "A"
    assert parse_judge("我觉得都不错").winner == "tie"
    assert parse_judge("").winner == "tie"


class FakeJudgeLLM:
    """按调用顺序返回固定结论，模拟位置偏差测试。"""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def chat(self, messages, **kwargs):
        from wagent.llm.client import ChatResult

        self.calls += 1
        text = self.answers.pop(0) if self.answers else '{"winner":"tie"}'
        return ChatResult(text, "fake", "fake", 1, 1, 0.0, 1)


def test_judge_tie_on_position_bias() -> None:
    # 第一次选 A，交换后仍选 A（实际是同一个回复）→ 判平局
    judge = LLMJudge(FakeJudgeLLM(['{"winner":"A"}', '{"winner":"A"}']))
    result = judge.compare("上下文", "回复一", "回复二")
    assert result.winner == "tie"


def test_judge_agrees_after_swap() -> None:
    # 第一次选 A（回复一），交换后选 B（也是回复一）→ 判定 A 胜
    judge = LLMJudge(FakeJudgeLLM(['{"winner":"A"}', '{"winner":"B"}']))
    assert judge.compare("上下文", "回复一", "回复二").winner == "A"


# ---------------------------------------------------------------- A/B


def test_run_ab_with_fake_judge() -> None:
    prompts = [
        ABPrompt(context_text="对方: 在吗", latest="在吗", reference="在的"),
        ABPrompt(context_text="对方: 有空吗", latest="有空吗", reference="有啊 咋了"),
    ]

    def gen_base(_prompt: ABPrompt) -> str:
        return "首先，我在这里。"

    def gen_tuned(_prompt: ABPrompt) -> str:
        return "在的"

    judge = LLMJudge(FakeJudgeLLM(['{"winner":"B"}', '{"winner":"A"}',
                                   '{"winner":"B"}', '{"winner":"A"}']))
    result = run_ab(prompts, gen_base, gen_tuned, judge=judge, references=REFS)
    assert result.n == 2
    assert result.b_wins == 2 and result.a_wins == 0
    assert result.win_rate("b") == 1.0
    assert result.tuned_metrics["score"] > result.base_metrics["score"]
    assert len(result.details) == 2


def test_run_ab_without_judge_and_limit() -> None:
    prompts = [ABPrompt(context_text=f"ctx{i}", latest=f"m{i}") for i in range(5)]
    result = run_ab(prompts, lambda _p: "行吧", lambda _p: "行的", limit=3)
    assert result.n == 3
    assert result.a_wins == 0 and result.b_wins == 0 and result.ties == 3
    assert result.win_rate("b") == 0.0


def test_run_ab_report_dict_serializable() -> None:
    import json

    prompts = [ABPrompt(context_text="ctx", latest="在吗")]
    result = run_ab(prompts, lambda _p: "行", lambda _p: "行吧")
    payload = json.dumps(result.as_dict(), ensure_ascii=False)
    assert "base_metrics" in payload
