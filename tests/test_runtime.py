"""运行时单测：感知 → 状态机 → 生成 → 发送 的整条链路（全部依赖注入）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from wagent.actuator import SendMode, SendPlan, SendResult
from wagent.agent.runtime import AgentRuntime, LLMGenerator, TickResult
from wagent.agent.state import ConversationState, Phase
from wagent.im.base import Frame
from wagent.im.mock import MockAdapter
from wagent.llm.client import ChatResult, LLMError
from wagent.perception.types import Box, Message, Speaker


def other(text: str) -> Message:
    return Message(speaker=Speaker.OTHER, text=text, box=Box(0, 0, 10, 10))


def me(text: str) -> Message:
    return Message(speaker=Speaker.ME, text=text, box=Box(0, 0, 10, 10))


def frame(*messages: Message, session: str = "张三") -> Frame:
    return Frame(messages=list(messages), session=session)


@dataclass
class FakeSender:
    """只记录调用，不碰系统输入。"""

    result: SendResult | None = None
    calls: list[tuple] = field(default_factory=list)

    def send(self, text, *, target_hwnd, click_point, mode):
        self.calls.append((text, target_hwnd, click_point, mode))
        if self.result is not None:
            return self.result
        plan = SendPlan(text=text, estimated_typing_s=1.0, think_delay_s=1.0, chunks=(text,))
        if mode is SendMode.DRY_RUN:
            return SendResult(False, "dry_run", 0.0, plan)
        return SendResult(True, "sent", 0.1, plan)


def make_runtime(
    frames: list[Frame],
    *,
    reply: str = "在的",
    mode: SendMode = SendMode.DRY_RUN,
    sender: FakeSender | None = None,
    generate=None,
) -> tuple[AgentRuntime, FakeSender, ConversationState]:
    sender = sender or FakeSender()
    state = ConversationState(cooldown_s=0.0)
    runtime = AgentRuntime(
        adapter=MockAdapter(frames),
        state=state,
        sender=sender,  # type: ignore[arg-type]
        generate=generate or (lambda trigger: reply),
        mode=mode,
    )
    return runtime, sender, state


# ---------------------------------------------------------------- 基本链路


def test_first_frame_is_baseline() -> None:
    runtime, sender, state = make_runtime([frame(other("在吗"))])
    result = runtime.tick()
    assert result.action == "no_trigger"
    assert sender.calls == []
    assert state.phase is Phase.IDLE


def test_new_message_preview_in_dry_run() -> None:
    runtime, sender, state = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))]
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "preview"
    assert result.reply == "在的"
    assert sender.calls and sender.calls[0][3] is SendMode.DRY_RUN
    assert state.phase is Phase.IDLE  # dry_run 后回到等待，避免卡在"已回复"
    assert state.skips == 1


def test_send_mode_marks_replying_until_echo() -> None:
    runtime, _sender, state = make_runtime(
        [
            frame(other("在吗")),
            frame(other("在吗"), other("今晚有空吗")),
            frame(other("今晚有空吗"), me("在的")),
        ],
        mode=SendMode.SEND,
    )
    runtime.tick()
    assert runtime.tick().action == "replied"
    assert state.phase is Phase.REPLYING
    # 自己的回复出现在屏幕上 → 进入冷却
    assert runtime.tick().action == "no_trigger"
    assert state.phase is Phase.COOLDOWN


def test_runtime_returns_no_trigger_when_frame_static() -> None:
    runtime, sender, _ = make_runtime([frame(other("在吗")), frame(other("在吗"))])
    runtime.tick()
    assert runtime.tick().action == "no_trigger"
    assert sender.calls == []


# ---------------------------------------------------------------- 失败处理


def test_generation_failure_does_not_crash_and_can_retry() -> None:
    calls = {"n": 0}

    def boom(_trigger) -> str:
        calls["n"] += 1
        raise LLMError("模型挂了")

    runtime, sender, state = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))], generate=boom
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "failed"
    assert "模型挂了" in result.error
    assert sender.calls == []
    assert state.phase is Phase.IDLE
    # fail() 清空指纹后允许下一轮重试同一条消息
    assert runtime.tick().action == "failed"
    assert calls["n"] == 2


def test_generation_returns_empty_skips() -> None:
    runtime, sender, _state = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))],
        generate=lambda _t: "   ",
    )
    runtime.tick()
    assert runtime.tick().action == "skipped"
    assert sender.calls == []


def test_input_box_missing_fails() -> None:
    class NoInputAdapter(MockAdapter):
        def input_box(self):
            return None

    sender = FakeSender()
    state = ConversationState()
    runtime = AgentRuntime(
        adapter=NoInputAdapter([frame(other("在吗")), frame(other("在吗"), other("新消息"))]),
        state=state,
        sender=sender,  # type: ignore[arg-type]
        generate=lambda _t: "在的",
    )
    runtime.tick()
    assert runtime.tick().action == "failed"
    assert sender.calls == []


def test_send_failure_marks_failed_state() -> None:
    sender = FakeSender(result=SendResult(False, "not_foreground", 0.0,
                                         SendPlan("在的", 1.0, 1.0, ("在的",))))
    runtime, _, state = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))],
        mode=SendMode.SEND,
        sender=sender,
    )
    runtime.tick()
    assert runtime.tick().action == "failed"
    assert state.phase is Phase.IDLE


def test_capture_exception_returns_failed() -> None:
    class BrokenAdapter(MockAdapter):
        def capture(self):
            raise RuntimeError("截图失败")

    runtime = AgentRuntime(
        adapter=BrokenAdapter([frame(other("在吗"))]),
        state=ConversationState(),
        sender=FakeSender(),  # type: ignore[arg-type]
        generate=lambda _t: "在的",
    )
    result = runtime.tick()
    assert result.action == "failed"
    assert "感知失败" in result.error


# ---------------------------------------------------------------- 事件回调 / LLMGenerator


def test_on_event_called_for_every_tick() -> None:
    events: list[TickResult] = []
    runtime, _, _ = make_runtime([frame(other("在吗")), frame(other("在吗"), other("喂"))])
    runtime.on_event = events.append
    runtime.tick()
    runtime.tick()
    assert len(events) == 2
    assert events[1].action == "preview"


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return ChatResult(self.text, "fake", "fake", 1, 1, 0.0, 1)


def test_llm_generator_builds_prompt_and_cleans_output() -> None:
    llm = FakeLLM("「在的。」")
    generator = LLMGenerator(llm, profile="后端工程师", fewshot=["行吧"])
    runtime, _, _ = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))],
        generate=generator,
    )
    runtime.tick()
    result = runtime.tick()
    assert result.action == "preview"
    assert result.reply == "在的"
    assert llm.messages is not None
    assert "后端工程师" in llm.messages[0].content
    assert "对方: 在吗" in llm.messages[1].content
