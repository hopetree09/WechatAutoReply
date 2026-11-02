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
from datetime import datetime

from wagent import config as config_module
from wagent.actuator import SafeSender, SendMode, TypingProfile
from wagent.agent.intent import IntentRouter, needs_memory
from wagent.agent.runtime import AgentRuntime, ChatModel, LLMGenerator, TickResult
from wagent.agent.state import ConversationState
from wagent.guard import check_output
from wagent.im.base import Frame, IMAdapter
from wagent.im.mock import MockAdapter
from wagent.llm.client import ChatResult, LLMClient
from wagent.llm.prompt import load_fewshot, load_profile
from wagent.memory import MemoryQuery, MemoryRetriever, MemoryStore, VectorIndex
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
    """一段脚本化对话：基线 → 新消息 → 自己的回复出现 → 语气词（跳过）→ 新消息。"""
    other, me = Speaker.OTHER, Speaker.ME
    session = "路旁的树"
    return [
        Frame([_msg(other, "在吗")], session),
        Frame([_msg(other, "在吗"), _msg(other, "今晚有空吗")], session),
        Frame([_msg(other, "在吗"), _msg(other, "今晚有空吗"), _msg(me, FAKE_REPLY)], session),
        Frame([_msg(other, "今晚有空吗"), _msg(me, FAKE_REPLY), _msg(other, "嗯")], session),
        Frame([_msg(me, FAKE_REPLY), _msg(other, "嗯"), _msg(other, "那就八点")], session),
    ]


# ---------------------------------------------------------------- 构建


def _review_confirm(text: str) -> bool:
    print(f"\n[待发送] {text}")
    try:
        answer = input("发送？[y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _seed_demo_memory(store: MemoryStore) -> None:
    """离线演示用的记忆数据（不碰用户真实数据库）。"""
    from wagent.memory import FactCard, PersonProfile, StyleSample
    from wagent.memory.types import SELF_PERSON

    store.upsert_profile(
        PersonProfile(person="路旁的树", display_name="路旁的树", relation="朋友",
                      traits=("爱聊技术",), preferences=("咖啡",))
    )
    store.upsert_profile(PersonProfile(person=SELF_PERSON, display_name="我", traits=("话少", "偶尔冷幽默")))
    store.add_fact(
        FactCard(person="路旁的树", subject="路旁的树", predicate="约了见面",
                 value="周三晚上八点老地方", confidence=0.9, evidence_text="今晚有空吗")
    )
    store.add_style_samples([StyleSample(text="行吧 先这样"), StyleSample(text="问题不大")])


def build_runtime(args: argparse.Namespace, mode: SendMode, tracer=None) -> AgentRuntime:
    settings = config_module.get_settings()

    adapter: IMAdapter
    if args.mock:
        adapter = MockAdapter(build_mock_frames())
    else:
        from wagent.im.wechat import WeChatAdapter

        adapter = WeChatAdapter(settings)

    llm: ChatModel = FakeLLM() if args.fake_llm else LLMClient(settings.llm)

    # 层级记忆：SQLite + 向量索引（本地 bge-m3；mock 模式用确定性 HashingEmbedder）
    memory_store = MemoryStore(":memory:") if args.mock else MemoryStore(settings.paths.db_path)
    if args.mock:
        _seed_demo_memory(memory_store)
    embedder = _build_embedder(mock=args.mock)
    vector_index = VectorIndex(memory_store)
    if embedder is not None:
        try:
            indexed = vector_index.sync(embedder)
            if any(indexed.values()):
                print(f"[memory] 向量索引：{indexed}")
        except Exception as exc:
            print(f"[memory] 向量索引失败（退回纯结构化检索）：{exc}")
    retriever = MemoryRetriever(memory_store, vector_index=vector_index, embedder=embedder)

    def memory_for(trigger) -> str:
        query = MemoryQuery(
            person=trigger.session or "unknown",
            latest_text=trigger.message.text,
            now=datetime.now().astimezone(),
        )
        return retriever.build(query).to_prompt_block()

    generator = LLMGenerator(
        llm,
        profile=load_profile(settings.paths.profile_path),
        fewshot=load_fewshot(settings.paths.fewshot_dir),
        memory_provider=memory_for,
        memory_policy=needs_memory,
    )

    typing = TypingProfile(**settings.typing.model_dump(exclude={"send_key"}))
    sender = SafeSender(
        typing,
        send_key=settings.typing.send_key,
        require_foreground=settings.agent.require_foreground,
        confirm=_review_confirm,
    )
    # 意图路由与安全护栏（默认开启；--no-intent / --no-guard 可关）
    intent_router = None
    if not getattr(args, "no_intent", False):
        use_llm = bool(getattr(args, "llm_intent", False))
        intent_router = IntentRouter(llm=llm if use_llm else None, use_llm=use_llm)
    guard = None
    if not getattr(args, "no_guard", False):

        def guard(reply: str, contexts: tuple[str, ...]):
            return check_output(reply, context_texts=contexts)

    return AgentRuntime(
        adapter=adapter,
        state=ConversationState(),
        sender=sender,
        generate=generator,
        mode=mode,
        tracer=tracer,
        intent_router=intent_router,
        guard=guard,
    )


def _build_embedder(*, mock: bool):
    """mock 用确定性本地 embedder；真实模式用本地 bge-m3（失败则降级）。"""
    from wagent.memory.embeddings import HashingEmbedder, OllamaEmbedder

    if mock:
        return HashingEmbedder(dim=64)
    try:
        return OllamaEmbedder()
    except Exception:
        return None


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
    parser.add_argument("--trace", action="store_true", help="把每轮耗时/动作写入 data/trace/*.jsonl")
    parser.add_argument("--no-intent", action="store_true", help="关闭意图路由（默认开启规则版）")
    parser.add_argument("--llm-intent", action="store_true", help="意图路由叠加 LLM 兜底（慢一点）")
    parser.add_argument("--no-guard", action="store_true", help="关闭输出安全护栏（默认开启）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = config_module.get_settings()
    mode = SendMode(args.mode) if args.mode is not None else SendMode(settings.agent.mode)

    tracer = None
    if args.trace:
        from wagent.observability import Tracer

        trace_path = settings.paths.data_dir / "trace" / f"trace-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        tracer = Tracer(trace_path)
        print(f"[trace] {trace_path}")

    try:
        runtime = build_runtime(args, mode, tracer=tracer)
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
    if tracer is not None:
        tracer.close()
        print(f"[trace] 汇总：{tracer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
