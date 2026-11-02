"""运行时安全链路单测：意图路由 + 输出护栏在 tick 里的行为。"""

from __future__ import annotations

from dataclasses import dataclass, field

from wagent.actuator import SendMode, SendPlan, SendResult
from wagent.agent.intent import Intent, IntentResult, IntentRouter
from wagent.agent.runtime import AgentRuntime
from wagent.agent.state import ConversationState, Phase
from wagent.guard import check_output
from wagent.im.base import Frame
from wagent.im.mock import MockAdapter
from wagent.perception.types import Box, Message, Speaker


def other(text: str) -> Message:
    return Message(speaker=Speaker.OTHER, text=text, box=Box(0, 0, 10, 10))


def frame(*messages: Message) -> Frame:
    return Frame(messages=list(messages), session="张三")


@dataclass
class FakeSender:
    calls: list[tuple] = field(default_factory=list)

    def send(self, text, *, target_hwnd, click_point, mode):
        self.calls.append(text)
        plan = SendPlan(text, 1.0, 1.0, (text,))
        return SendResult(True, "sent", 0.1, plan)


class StaticRouter:
    def __init__(self, intent: Intent, confidence: float = 0.9) -> None:
        self.result = IntentResult(intent, confidence, "test")

    def classify(self, text: str) -> IntentResult:
        return self.result


def make_runtime(frames, *, reply="在的", router=None, guard=None):
    sender = FakeSender()
    runtime = AgentRuntime(
        adapter=MockAdapter(frames),
        state=ConversationState(cooldown_s=0.0),
        sender=sender,  # type: ignore[arg-type]
        generate=lambda _t: reply,
        mode=SendMode.SEND,
        intent_router=router,
        guard=guard,
    )
    return runtime, sender


def run_trigger(runtime) -> None:
    """第一帧建基线，第二帧制造新消息 → 触发。"""
    runtime.tick()
    runtime.tick()


# ---------------------------------------------------------------- 意图路由


def test_no_reply_intent_skips_generation_and_send() -> None:
    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("嗯"))],
        router=StaticRouter(Intent.NO_REPLY),
    )
    run_trigger(runtime)
    runtime.tick()
    # 第二次 tick 已经处理完；这里直接看第一次触发的行为
    assert sender.calls == []


def test_no_reply_intent_returns_skipped_with_note() -> None:
    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("嗯"))],
        router=StaticRouter(Intent.NO_REPLY),
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "skipped"
    assert "无需回复" in result.note
    assert sender.calls == []
    assert runtime.state.phase is Phase.IDLE


def test_sensitive_intent_skips_reply() -> None:
    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("借我500块"))],
        router=StaticRouter(Intent.SENSITIVE),
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "skipped"
    assert "敏感话题" in result.note
    assert sender.calls == []


def test_plan_intent_proceeds_to_generate() -> None:
    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("周末有空吗"))],
        router=StaticRouter(Intent.PLAN),
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "replied"
    assert sender.calls == ["在的"]


# ---------------------------------------------------------------- 护栏


def test_guard_blocks_money_reply_before_sending() -> None:
    def guard(text, contexts):
        return check_output(text, context_texts=contexts)

    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("帮我付一下"))],
        reply="好的我转你 200 块",
        guard=guard,
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "skipped"
    assert "金钱" in result.note
    assert sender.calls == []


def test_guard_masks_pii_before_sending() -> None:
    def guard(text, contexts):
        return check_output(text, context_texts=contexts)

    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("怎么注册"))],
        reply="用这个手机号 13812345678 注册",
        guard=guard,
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "replied"
    assert sender.calls == ["用这个手机号 ***手机号*** 注册"]
    assert "13812345678" not in sender.calls[0]


def test_guard_blocks_verbatim_repetition() -> None:
    def guard(text, contexts):
        return check_output(text, context_texts=contexts)

    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))],
        reply="今晚有空吗",
        guard=guard,
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "skipped"
    assert sender.calls == []


def test_real_router_integration_on_word_only_message() -> None:
    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("嗯"))],
        router=IntentRouter(),
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "skipped"
    assert sender.calls == []


def test_without_router_and_guard_behaves_as_before() -> None:
    runtime, sender = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("嗯"))],
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "replied"
    assert sender.calls == ["在的"]
