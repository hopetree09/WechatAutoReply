"""可观测性单测：trace 事件、JSONL 落盘、运行时埋点。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from wagent.actuator import SendMode, SendPlan, SendResult
from wagent.agent.runtime import AgentRuntime
from wagent.agent.state import ConversationState
from wagent.im.base import Frame
from wagent.im.mock import MockAdapter
from wagent.observability import Tracer
from wagent.perception.types import Box, Message, Speaker


def other(text: str) -> Message:
    return Message(speaker=Speaker.OTHER, text=text, box=Box(0, 0, 10, 10))


def frame(*messages: Message) -> Frame:
    return Frame(messages=list(messages), session="张三")


@dataclass
class FakeSender:
    result: SendResult | None = None
    calls: list = field(default_factory=list)

    def send(self, text, *, target_hwnd, click_point, mode):
        self.calls.append(text)
        plan = SendPlan(text, 1.0, 1.0, (text,))
        return self.result or SendResult(False, "dry_run", 0.0, plan)


def make_runtime(frames, *, tracer=None, generate=None):
    return AgentRuntime(
        adapter=MockAdapter(frames),
        state=ConversationState(cooldown_s=0.0),
        sender=FakeSender(),  # type: ignore[arg-type]
        generate=generate or (lambda _t: "在的"),
        mode=SendMode.DRY_RUN,
        tracer=tracer,
    )


def test_tracer_events_and_summary() -> None:
    tracer = Tracer()
    tracer.event("perceive", ms=12.5, messages=3)
    tracer.event("generate", ms=800.0, reply="在的")
    tracer.event("error", stage="send", error="boom")
    summary = tracer.summary()
    assert summary["events"] == 3
    assert summary["counts"]["perceive"] == 1
    assert summary["total_ms"]["generate"] == 800.0
    assert summary["errors"] == 1


def test_tracer_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path)
    tracer.event("tick", action="no_trigger", ms=1.0)
    tracer.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["kind"] == "tick" and row["action"] == "no_trigger"
    assert row["seq"] == 1


def test_tracer_span_records_ms() -> None:
    import time

    tracer = Tracer()
    started = time.perf_counter()
    time.sleep(0.01)
    tracer.span("perceive", started, messages=1)
    assert tracer.events[0].data["ms"] > 5


def test_runtime_emits_expected_sequence() -> None:
    tracer = Tracer()
    runtime = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("今晚有空吗"))], tracer=tracer
    )
    runtime.tick()
    runtime.tick()
    kinds = [event.kind for event in tracer.events]
    assert "perceive" in kinds
    assert "trigger" in kinds
    assert "generate" in kinds
    assert "send" in kinds
    # 每个 tick 一个收尾事件
    assert len([k for k in kinds if k == "tick"]) == 2
    trigger_events = [e for e in tracer.events if e.kind == "trigger"]
    assert trigger_events[0].data["text"] == "今晚有空吗"
    send_events = [e for e in tracer.events if e.kind == "send"]
    assert send_events[0].data["reason"] == "dry_run"


def test_runtime_traces_perceive_error() -> None:
    class BrokenAdapter(MockAdapter):
        def capture(self):
            raise RuntimeError("截图失败")

    tracer = Tracer()
    runtime = AgentRuntime(
        adapter=BrokenAdapter([frame(other("在吗"))]),
        state=ConversationState(),
        sender=FakeSender(),  # type: ignore[arg-type]
        generate=lambda _t: "在的",
        tracer=tracer,
    )
    assert runtime.tick().action == "failed"
    errors = [event for event in tracer.events if event.kind == "error"]
    assert errors and errors[0].data["stage"] == "perceive"


def test_runtime_traces_generate_error() -> None:
    from wagent.llm.client import LLMError

    def boom(_trigger):
        raise LLMError("模型挂了")

    tracer = Tracer()
    runtime = make_runtime(
        [frame(other("在吗")), frame(other("在吗"), other("喂"))], tracer=tracer, generate=boom
    )
    runtime.tick()
    runtime.tick()
    errors = [event for event in tracer.events if event.kind == "error"]
    assert errors and errors[0].data["stage"] == "generate"


def test_runtime_without_tracer_still_works() -> None:
    runtime = make_runtime([frame(other("在吗")), frame(other("在吗"), other("喂"))])
    runtime.tick()
    assert runtime.tick().action == "preview"
