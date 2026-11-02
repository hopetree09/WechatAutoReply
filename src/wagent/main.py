"""命令行入口。

用法::

    # 1) 离线演示（不碰微信、不联网，用假 LLM + 脚本化对话）
    python -m wagent.main --mock

    # 2) 真机只读观察：截图 + OCR + 生成，但绝不发送（默认 dry_run）
    python -m wagent.main

    # 3) 人工确认后发送
    python -m wagent.main --mode review

    # 4) 全自动（危险，需自行确认）
    python -m wagent.main --mode send
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from wagent import config as config_module
from wagent.actuator import SafeSender, SendMode, TypingProfile
from wagent.agent.runtime import AgentRuntime, ChatModel, LLMGenerator, TickResult
from wagent.agent.state import ConversationState
from wagent.im.base import Frame, IMAdapter
from wagent.im.mock import MockAdapter
from wagent.llm.client import ChatResult, LLMClient
from wagent.llm.prompt import load_fewshot, load_profile
from wagent.perception.types import Box, Message, Speaker

FAKE_REPLY = "在的 八点老地方见"


# ---------------------------------------------------------------- 假 LLM / demo 脚本


class FakeLLM:
    """不联网的假模型，用于 --fake-llm 与离线演示。"""

    def __init__(self, reply: str = FAKE_REPLY) -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, messages, **kwargs) -> ChatResult:
        self.calls += 1
        return ChatResult(
            text=self.reply,
            model="fake",
            provider="fake",
            tokens_in=0,
            tokens_out=0,
            latency_s=0.0,
            attempts=1,
        )


def _msg(speaker: Speaker, text: str) -> Message:
    return Message(speaker=speaker, text=text, box=Box(0, 0, 10, 10))


def build_mock_frames() -> list[Frame]:
    """一段脚本化对话：基线 → 新消息 → 自己的回复出现 → 对方再发一条。"""
    other, me = Speaker.OTHER, Speaker.ME
    session = "路旁的树"
    return [
        Frame([_msg(other, "在吗")], session),
        Frame([_msg(other, "在吗"), _msg(other, "今晚有空吗")], session),
        Frame([_msg(other, "在吗"), _msg(other, "今晚有空吗"), _msg(me, FAKE_REPLY)], session),
        Frame([_msg(other, "今晚有空吗"), _msg(me, FAKE_REPLY), _msg(other, "那就八点")], session),
    ]


# ---------------------------------------------------------------- 构建


def _review_confirm(text: str) -> bool:
    print(f"\n[待发送] {text}")
    try:
        answer = input("发送？[y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def build_runtime(args: argparse.Namespace, mode: SendMode) -> AgentRuntime:
    settings = config_module.get_settings()

    adapter: IMAdapter
    if args.mock:
        adapter = MockAdapter(build_mock_frames())
    else:
        from wagent.im.wechat import WeChatAdapter

        adapter = WeChatAdapter(settings)

    llm: ChatModel = FakeLLM() if args.fake_llm else LLMClient(settings.llm)
    generator = LLMGenerator(
        llm,
        profile=load_profile(settings.paths.profile_path),
        fewshot=load_fewshot(settings.paths.fewshot_dir),
    )

    typing = TypingProfile(**settings.typing.model_dump(exclude={"send_key"}))
    sender = SafeSender(
        typing,
        send_key=settings.typing.send_key,
        require_foreground=settings.agent.require_foreground,
        confirm=_review_confirm,
    )
    return AgentRuntime(
        adapter=adapter,
        state=ConversationState(),
        sender=sender,
        generate=generator,
        mode=mode,
    )


# ---------------------------------------------------------------- 主循环


def format_tick(index: int, result: TickResult) -> str:
    stamp = time.strftime("%H:%M:%S")
    return f"[{stamp}] #{index:<3} {result.describe()}"


def run(
    runtime: AgentRuntime,
    *,
    interval_s: float,
    max_ticks: int | None = None,
    printer=print,
) -> list[TickResult]:
    results: list[TickResult] = []
    index = 0
    try:
        while max_ticks is None or index < max_ticks:
            index += 1
            result = runtime.tick()
            results.append(result)
            printer(format_tick(index, result))
            if (max_ticks is None or index < max_ticks) and interval_s > 0:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        printer("\n[*] 收到 Ctrl+C，已退出")
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wagent：桌面微信 GUI Agent")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in SendMode],
        default=None,
        help="dry_run（默认，只预览）/ review（人工确认）/ send（直接发送）",
    )
    parser.add_argument("--mock", action="store_true", help="离线演示：不碰微信")
    parser.add_argument("--fake-llm", action="store_true", help="不联网的假模型")
    parser.add_argument("--once", action="store_true", help="只跑一轮")
    parser.add_argument("--ticks", type=int, default=0, help="跑固定轮数后退出（0=无限）")
    parser.add_argument("--interval", type=float, default=None, help="轮询间隔（秒）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = config_module.get_settings()
    mode = SendMode(args.mode) if args.mode is not None else SendMode(settings.agent.mode)

    try:
        runtime = build_runtime(args, mode)
    except Exception as exc:
        print(f"[!] 初始化失败: {exc}")
        return 2

    interval = settings.agent.poll_interval_s if args.interval is None else args.interval
    max_ticks: int | None
    if args.mock:
        max_ticks = len(build_mock_frames()) + 1
        interval = 0.0
    elif args.once:
        max_ticks = 1
    elif args.ticks > 0:
        max_ticks = args.ticks
    else:
        max_ticks = None

    backend = "mock" if args.mock else "wechat"
    model = "fake" if args.fake_llm else f"{settings.llm.provider}:{settings.llm.local_model if settings.llm.provider == 'local' else settings.llm.cloud_model}"
    print(f"[*] backend={backend} model={model} mode={mode.value} interval={interval}s")
    if mode is not SendMode.DRY_RUN:
        print("[!] 非 dry_run 模式会真的操作键鼠，请确认微信窗口位置正确")

    run(runtime, interval_s=interval, max_ticks=max_ticks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
