"""主循环运行时：感知 → 触发状态机 → 生成 → 安全发送。

把「什么时候回」「回什么」「怎么发」三件事解耦：
- 状态机 :class:`wagent.agent.state.ConversationState` 决定时机；
- ``generate`` 回调决定内容（真实场景是 :class:`LLMGenerator`）；
- :class:`wagent.actuator.SafeSender` 决定怎么发，并负责安全校验。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from wagent.actuator import SafeSender, SendMode, SendResult
from wagent.agent.state import ConversationState, Trigger
from wagent.im.base import IMAdapter
from wagent.llm.client import ChatModel, ChatResult, LLMError
from wagent.llm.prompt import build_messages, clean_reply

Action = Literal["no_trigger", "preview", "replied", "skipped", "failed"]

Generator = Callable[[Trigger], str]

__all__ = [
    "AgentRuntime",
    "ChatModel",
    "ChatResult",
    "Generator",
    "LLMGenerator",
    "TickResult",
]


@dataclass(slots=True)
class TickResult:
    """一次循环的结果（用于打印/日志/评测）。"""

    action: Action
    trigger: Trigger | None = None
    reply: str = ""
    send: SendResult | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.action != "failed"

    def describe(self) -> str:
        if self.action == "no_trigger":
            return "无新消息"
        if self.action == "preview":
            return f"[预览·未发送] {self.reply}"
        if self.action == "replied":
            elapsed = f"（{self.send.elapsed_s:.1f}s）" if self.send else ""
            return f"[已发送{elapsed}] {self.reply}"
        if self.action == "skipped":
            reason = self.send.reason if self.send else ""
            return f"[跳过] {self.reply or reason}"
        return f"[失败] {self.error}"


class LLMGenerator:
    """把 Trigger 变成回复文本（可注入层级记忆检索）。"""

    def __init__(
        self,
        llm: ChatModel,
        *,
        profile: str = "",
        fewshot: Sequence[str] = (),
        limit: int = 12,
        memory_provider: Callable[[Trigger], str] | None = None,
    ) -> None:
        self.llm = llm
        self.profile = profile
        self.fewshot = list(fewshot)
        self.limit = limit
        self.memory_provider = memory_provider

    def _memory_block(self, trigger: Trigger) -> str:
        if self.memory_provider is None:
            return ""
        try:
            return self.memory_provider(trigger) or ""
        except Exception:
            return ""

    def __call__(self, trigger: Trigger) -> str:
        turns = build_messages(
            trigger.context,
            trigger.message.text,
            profile=self.profile,
            fewshot=self.fewshot,
            limit=self.limit,
            memory_block=self._memory_block(trigger),
        )
        result = self.llm.chat(turns)
        return clean_reply(result.text)


@dataclass
class AgentRuntime:
    """串联感知/决策/生成的运行时。"""

    adapter: IMAdapter
    state: ConversationState
    sender: SafeSender
    generate: Generator
    mode: SendMode = SendMode.DRY_RUN
    on_event: Callable[[TickResult], None] | None = field(default=None, repr=False)

    def tick(self) -> TickResult:
        """跑一轮；任何异常都收敛成 ``failed``，不让主循环崩掉。"""
        try:
            frame = self.adapter.capture()
        except Exception as exc:
            return self._emit(TickResult("failed", error=f"感知失败: {exc}"))

        trigger = self.state.observe(frame.messages, session=frame.session)
        if trigger is None:
            return self._emit(TickResult("no_trigger"))

        try:
            reply = self.generate(trigger)
        except LLMError as exc:
            self.state.fail()
            return self._emit(TickResult("failed", trigger=trigger, error=f"生成失败: {exc}"))
        except Exception as exc:
            self.state.fail()
            return self._emit(TickResult("failed", trigger=trigger, error=f"生成异常: {exc}"))

        if not reply.strip():
            self.state.skip()
            return self._emit(TickResult("skipped", trigger=trigger, reply=""))

        input_box = self.adapter.input_box()
        if input_box is None:
            self.state.fail()
            return self._emit(TickResult("failed", trigger=trigger, error="未定位到输入框"))

        try:
            send_result = self.sender.send(
                reply,
                target_hwnd=self.adapter.hwnd,
                click_point=input_box.click_point(),
                mode=self.mode,
            )
        except Exception as exc:
            self.state.fail()
            return self._emit(
                TickResult("failed", trigger=trigger, reply=reply, error=f"发送异常: {exc}")
            )

        if send_result.sent:
            self.state.mark_sent(reply)
            return self._emit(
                TickResult("replied", trigger=trigger, reply=reply, send=send_result)
            )
        if send_result.reason == "dry_run":
            self.state.skip()
            return self._emit(
                TickResult("preview", trigger=trigger, reply=reply, send=send_result)
            )
        if send_result.reason == "review_rejected":
            self.state.skip()
            return self._emit(
                TickResult("skipped", trigger=trigger, reply=reply, send=send_result)
            )
        self.state.fail()
        return self._emit(
            TickResult(
                "failed",
                trigger=trigger,
                reply=reply,
                send=send_result,
                error=send_result.reason,
            )
        )

    def _emit(self, result: TickResult) -> TickResult:
        if self.on_event is not None:
            self.on_event(result)
        return result
