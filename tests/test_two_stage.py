"""两段式生成（7B 草稿 → 3B 改写）单测：prompt 构造、降级、记忆策略。"""

from __future__ import annotations

from types import SimpleNamespace

from wagent.agent.intent import needs_memory
from wagent.agent.state import Trigger
from wagent.agent.two_stage import TwoStageGenerator, build_rewrite_messages
from wagent.perception.types import Box, Message, Speaker


def message(speaker: str, text: str) -> Message:
    return Message(
        speaker=Speaker.ME if speaker == "me" else Speaker.OTHER,
        text=text,
        box=Box(0, 0, 1, 1),
    )


def trigger(text: str, context: list[Message] | None = None) -> Trigger:
    return Trigger(
        message=message("other", text),
        context=context or [message("me", "在吗"), message("other", "嗯")],
        session="吴浩毅",
    )


class FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages, **kwargs):
        self.prompts.append("\n".join(getattr(m, "content", str(m)) for m in messages))
        reply = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(text=reply)


def test_build_rewrite_messages_contains_draft_and_latest() -> None:
    turns = build_rewrite_messages([message("me", "在吗")], "周末吃火锅不", "可以啊 你定地点")
    assert len(turns) == 2
    assert "改写器" in turns[0].content
    assert "周末吃火锅不" in turns[1].content
    assert "可以啊 你定地点" in turns[1].content


def test_two_stage_uses_second_llm_output() -> None:
    draft = FakeLLM(["7B 草稿：可以的，你定地方吧"])
    rewrite = FakeLLM(["行 你定"])
    generator = TwoStageGenerator(draft_llm=draft, rewrite_llm=rewrite, min_fidelity=0.0)
    assert generator(trigger("周末吃火锅不")) == "行 你定"
    assert "7B 草稿" in rewrite.prompts[0]


def test_two_stage_falls_back_when_rewrite_loses_content() -> None:
    draft_text = "行 周日下午我回来了 周边有啥好吃的推荐不"
    generator = TwoStageGenerator(
        draft_llm=SimpleNamespace(chat=lambda _m: SimpleNamespace(text=draft_text)),
        rewrite_llm=FakeLLM(["有的"]),
    )
    assert generator(trigger("周末吃火锅不")) == draft_text


def test_two_stage_accepts_rewrite_when_content_preserved() -> None:
    draft_text = "行 周日下午我回来了 周边有啥好吃的推荐不"
    generator = TwoStageGenerator(
        draft_llm=SimpleNamespace(chat=lambda _m: SimpleNamespace(text=draft_text)),
        rewrite_llm=FakeLLM(["行 周日下午回来 有啥好吃的推荐"]),
    )
    assert generator(trigger("周末吃火锅不")) == "行 周日下午回来 有啥好吃的推荐"


def test_two_stage_memory_policy() -> None:
    drafts: list[str] = []

    def draft(messages):
        drafts.append("\n".join(getattr(m, "content", "") for m in messages))
        return SimpleNamespace(text="草稿")

    memory = "【我记得的事】\n- 我的显卡是 5060"
    generator = TwoStageGenerator(
        draft_llm=SimpleNamespace(chat=draft),
        rewrite_llm=FakeLLM(["好"]),
        memory_provider=lambda _t: memory,
        memory_policy=needs_memory,
    )
    generator(trigger("嗯"))
    assert "显卡" not in drafts[-1]
    generator(trigger("你电脑什么配置"))
    assert "显卡" in drafts[-1]


def test_two_stage_falls_back_to_draft_when_rewrite_fails() -> None:
    class Broken:
        def chat(self, messages, **kwargs):
            raise RuntimeError("3B 挂了")

    generator = TwoStageGenerator(
        draft_llm=SimpleNamespace(chat=lambda _m: SimpleNamespace(text="7B 草稿")),
        rewrite_llm=Broken(),
    )
    assert generator(trigger("在吗")) == "7B 草稿"


def test_two_stage_empty_draft_returns_empty() -> None:
    generator = TwoStageGenerator(
        draft_llm=SimpleNamespace(chat=lambda _m: SimpleNamespace(text="")),
        rewrite_llm=FakeLLM(["不会被用到"]),
    )
    assert generator(trigger("在吗")) == ""
