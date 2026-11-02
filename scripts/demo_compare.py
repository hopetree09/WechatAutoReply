"""不碰微信的自测：用手写上下文对比三组 prompt + 跑一遍完整运行时。

1) 「裸模型 / +自动画像 / +画像+层级记忆」三组回复对比（上下文由脚本给出，不读取任何真实聊天）；
2) 用 MockAdapter 驱动真实 AgentRuntime（真实画像 + 真实记忆库），验证：
   - 语气词 → 意图路由跳过；
   - 借钱 → 敏感话题跳过；
   - 「你电脑什么配置」→ 记忆命中后回复；
   - 正常消息 → 预览（dry_run 不发送）。

用法：
    python scripts/demo_compare.py            # 全部
    python scripts/demo_compare.py --no-judge # 跳过 LLM 盲评
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.actuator import SafeSender, SendMode, SendPlan, SendResult, TypingProfile  # noqa: E402
from wagent.agent.intent import IntentRouter, needs_memory  # noqa: E402
from wagent.agent.runtime import AgentRuntime, LLMGenerator  # noqa: E402
from wagent.agent.state import ConversationState  # noqa: E402
from wagent.config import get_settings  # noqa: E402
from wagent.eval.metrics import score_reply  # noqa: E402
from wagent.guard import check_output  # noqa: E402
from wagent.im.base import Frame  # noqa: E402
from wagent.im.mock import MockAdapter  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402
from wagent.llm.prompt import build_messages, clean_reply, load_fewshot, load_profile  # noqa: E402
from wagent.memory import MemoryQuery, MemoryRetriever, MemoryStore, VectorIndex  # noqa: E402
from wagent.memory.embeddings import OllamaEmbedder  # noqa: E402
from wagent.perception.types import Box, Message, Speaker  # noqa: E402

SESSION = "吴浩毅"  # 记忆库里的 person key；上下文是脚本手写的，不含真实聊天内容


@dataclass(frozen=True)
class DemoCase:
    topic: str
    context: tuple[tuple[str, str], ...]
    latest: str


@dataclass(frozen=True)
class Row:
    case: DemoCase
    a: str
    b: str
    c: str
    hit: bool


CASES = (
    DemoCase(
        "约饭",
        (("我", "刚开完组会"), ("对方", "累死了吧"), ("我", "还行 就是想躺")),
        "周末有空吗 出来吃火锅",
    ),
    DemoCase(
        "技术求助",
        (("对方", "在吗 问你个事"), ("我", "说")),
        "我 python 读文件报 KeyError 咋整啊",
    ),
    DemoCase(
        "回忆-电脑配置",
        (("对方", "你换电脑了？"), ("我", "换了")),
        "你电脑什么配置来着 显卡多少",
    ),
    DemoCase(
        "情绪安慰",
        (("对方", "今天又被导师说了"), ("我", "啊这")),
        "烦死了 不想干了",
    ),
    DemoCase(
        "玩笑互怼",
        (("对方", "最近咋样"), ("我", "老样子")),
        "你是不是又胖了",
    ),
    DemoCase(
        "热点讨论",
        (("对方", "刷到那个代言人的事没"), ("我", "刷到了")),
        "小红书这波处理你怎么看",
    ),
    DemoCase(
        "约时间",
        (("对方", "这周忙不"), ("我", "有点")),
        "那周末你回来不 一起吃饭",
    ),
    DemoCase(
        "纯语气词",
        (("对方", "我先去洗澡了"), ("我", "去吧")),
        "嗯",
    ),
)


def _msg(speaker: str, text: str) -> Message:
    return Message(
        speaker=Speaker.ME if speaker == "我" else Speaker.OTHER,
        text=text,
        box=Box(0, 0, 1, 1),
    )


def memory_block(retriever: MemoryRetriever, latest: str) -> str:
    query = MemoryQuery(person=SESSION, latest_text=latest, now=datetime.now().astimezone())
    return retriever.build(query).to_prompt_block()


def generate(llm, case: DemoCase, *, profile: str, fewshot: list[str], memory: str) -> str:
    context = [_msg(speaker, text) for speaker, text in case.context]
    turns = build_messages(context, case.latest, profile=profile, fewshot=fewshot, memory_block=memory)
    return clean_reply(llm.chat(turns).text)


def generate_short(llm, case: DemoCase) -> str:
    """LoRA 训练时的短 system prompt（原 adapter 的训练分布），用于验证「纯风格引擎」。"""
    from wagent.llm.client import MessageTurn
    from wagent.train.dataset import DEFAULT_SYSTEM

    history = "\n".join(
        f"{'我' if speaker == '我' else '对方'}: {text}" for speaker, text in case.context
    )
    user = (
        f"最近聊天记录：\n{history}\n\n"
        f"对方刚发来：{case.latest}\n\n"
        "请以我的口吻回复（只输出回复内容）："
    )
    turns = [MessageTurn("system", DEFAULT_SYSTEM), MessageTurn("user", user)]
    return clean_reply(llm.chat(turns).text)


class FakeSender(SafeSender):
    """记录发送调用，不碰键鼠。"""

    def __init__(self) -> None:
        super().__init__(TypingProfile(), require_foreground=False)
        self.calls: list[tuple[str, SendMode]] = []

    def send(
        self,
        text: str,
        *,
        target_hwnd: int,
        click_point: tuple[int, int],
        mode: SendMode = SendMode.DRY_RUN,
        confirm=None,
    ):
        self.calls.append((text, mode))
        plan = SendPlan(text=text, estimated_typing_s=0.0, think_delay_s=0.0, chunks=(text,))
        return SendResult(sent=True, reason="faked", elapsed_s=0.0, plan=plan)


def run_runtime_demo(llm, retriever, profile: str, fewshot: list[str]) -> None:
    """用 MockAdapter 跑完整 runtime（真实画像 + 真实记忆），验证决策链路。"""
    print("\n" + "#" * 72)
    print("# 完整运行时行为（MockAdapter，不碰微信）")
    print("#" * 72)

    def frame(messages: list[Message]) -> Frame:
        return Frame(messages=messages, session=SESSION)

    scenarios = {
        "语气词「嗯」": [
            frame([_msg("对方", "在吗")]),
            frame([_msg("对方", "在吗"), _msg("对方", "嗯")]),
        ],
        "借钱（敏感）": [
            frame([_msg("对方", "在吗")]),
            frame([_msg("对方", "在吗"), _msg("对方", "借我两千块周转一下")]),
        ],
        "电脑配置（记忆）": [
            frame([_msg("对方", "你换电脑了？")]),
            frame([_msg("对方", "你换电脑了？"), _msg("对方", "你电脑什么配置来着 显卡多少")]),
        ],
        "普通约饭": [
            frame([_msg("对方", "在吗")]),
            frame([_msg("对方", "在吗"), _msg("对方", "周末有空吗 出来吃火锅")]),
        ],
    }

    def memory_for(trigger) -> str:
        if not needs_memory(trigger.message.text):
            return ""
        query = MemoryQuery(
            person=trigger.session or SESSION,
            latest_text=trigger.message.text,
            now=datetime.now().astimezone(),
        )
        return retriever.build(query).to_prompt_block()

    generator = LLMGenerator(
        llm,
        profile=profile,
        fewshot=fewshot,
        limit=12,
        memory_provider=memory_for,
        memory_policy=needs_memory,
    )
    for name, frames in scenarios.items():
        sender = FakeSender()
        runtime = AgentRuntime(
            adapter=MockAdapter(frames),
            state=ConversationState(cooldown_s=0.0),
            sender=sender,
            generate=generator,
            mode=SendMode.DRY_RUN,
            intent_router=IntentRouter(),
            guard=lambda text, contexts: check_output(text, context_texts=contexts),
        )
        runtime.tick()  # 基线
        result = runtime.tick()  # 触发
        print(f"\n[{name}] {result.action}: {result.describe()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="不碰微信的 prompt / 运行时自测")
    parser.add_argument("--judge", action="store_true", help="跑 LLM 盲评（A vs C，用本地 7B）")
    parser.add_argument(
        "--provider",
        choices=("ollama", "lora"),
        default="ollama",
        help="生成模型：ollama=本地 7B（默认）；lora=Qwen2.5-3B + 真实数据 adapter",
    )
    parser.add_argument(
        "--base",
        default=str(ROOT / "models" / "models" / "Qwen--Qwen2.5-3B-Instruct" / "snapshots" / "master"),
    )
    parser.add_argument("--adapter", default=str(ROOT / "data" / "lora" / "qwen25-3b-real"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--system-mode",
        choices=("full", "short"),
        default="full",
        help="full=画像+记忆的长 system prompt；short=LoRA 训练时的短 system prompt",
    )
    parser.add_argument("--no-runtime", action="store_true", help="跳过 MockAdapter 运行时演示")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(settings.paths.profile_path)
    fewshot = load_fewshot(settings.paths.fewshot_dir)
    llm: Any
    if args.provider == "lora":
        from wagent.train.lora import LoraChatModel

        llm = LoraChatModel(
            base_model=args.base,
            adapter_dir=args.adapter,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[demo] 模型：Qwen2.5-3B + LoRA adapter={args.adapter}")
        print("[demo] 首次生成会加载 4bit 模型（约 20s）…")
    else:
        llm = LLMClient(settings.llm)
        print(f"[demo] 模型：Ollama {settings.llm.local_model}")

    store = MemoryStore(settings.paths.db_path)
    retriever = MemoryRetriever(store)
    try:
        embedder = OllamaEmbedder()
        vector_index = VectorIndex(store)
        vector_index.sync(embedder)
        retriever = MemoryRetriever(store, vector_index=vector_index, embedder=embedder)
    except Exception as exc:
        print(f"[demo] 向量检索不可用，退回关键词：{exc}")

    if args.system_mode == "short":
        print("=" * 72)
        print("LoRA 短 system prompt（训练分布）：每例只生成一条")
        print("=" * 72)
        short_rows = [(case, generate_short(llm, case)) for case in CASES]
        for case, reply in short_rows:
            score = score_reply(reply)
            print(f"\n[{case.topic}] 对方: {case.latest}")
            print(f"  LoRA: {reply}")
            print(f"    综合 {score.score:.3f} | AI味 {score.ai_flavor:.3f} | 长度 {len(reply)} 字")
        scores = [score_reply(reply) for _case, reply in short_rows]
        print(f"\n[short] 平均综合 {sum(s.score for s in scores) / len(scores):.3f} | "
              f"AI 味 {sum(s.ai_flavor for s in scores) / len(scores):.3f} | "
              f"平均长度 {sum(len(r) for _c, r in short_rows) / len(short_rows):.1f} 字")
        if not args.no_runtime:
            run_runtime_demo(llm, retriever, profile, fewshot)
        store.close()
        return 0

    rows = []
    print("=" * 72)
    print("三组对比：A 裸模型 / B +自动画像 / C +画像+层级记忆（上下文为脚本手写）")
    print("=" * 72)
    for case in CASES:
        inject = needs_memory(case.latest)
        mem = memory_block(retriever, case.latest) if inject else ""
        reply_a = generate(llm, case, profile="", fewshot=[], memory="")
        reply_b = generate(llm, case, profile=profile, fewshot=fewshot, memory="")
        reply_c = generate(llm, case, profile=profile, fewshot=fewshot, memory=mem)
        rows.append(Row(case=case, a=reply_a, b=reply_b, c=reply_c, hit=bool(mem.strip())))
        print(f"  [memory] 注入={bool(mem.strip())} needs_memory={inject}")
        print("\n" + "-" * 72)
        print(f"[{case.topic}] 对方: {case.latest}")
        print(f"  A 裸模型     : {reply_a}")
        print(f"  B +画像      : {reply_b}")
        print(f"  C +画像+记忆 : {reply_c}")
        for key, reply in (("A", reply_a), ("B", reply_b), ("C", reply_c)):
            score = score_reply(reply)
            print(f"    {key} 综合 {score.score:.3f} | AI味 {score.ai_flavor:.3f} | 长度 {len(reply)} 字")

    print("\n" + "=" * 72)
    print("汇总（无参考回复，只看自然度/AI 味/长度）")
    for key, attr in (("A 裸模型", "a"), ("B +画像", "b"), ("C +画像+记忆", "c")):
        replies = [getattr(row, attr) for row in rows]
        scores = [score_reply(reply) for reply in replies]
        print(f"{key:<12} 平均综合 {sum(s.score for s in scores) / len(scores):.3f} | "
              f"AI 味 {sum(s.ai_flavor for s in scores) / len(scores):.3f} | "
              f"平均长度 {sum(len(r) for r in replies) / len(replies):.1f} 字")
    hit = sum(1 for row in rows if row.hit)
    print(f"[demo] 记忆命中：{hit}/{len(rows)} 条上下文带记忆块")

    if args.judge:
        from wagent.eval.ab import LLMJudge

        # 评委固定用本地 7B（3B 4bit 与 7B 同时驻留容易 OOM）
        judge = LLMJudge(LLMClient(get_settings().llm))
        wins = {"A 裸模型": 0, "C +画像+记忆": 0, "平": 0}
        for row in rows:
            result = judge.compare(
                "\n".join(f"{who}: {text}" for who, text in row.case.context),
                row.a,
                row.c,
            )
            winner = "平" if result.winner == "tie" else ("C +画像+记忆" if result.winner == "b" else "A 裸模型")
            wins[winner] += 1
        print(f"[judge] A vs C：{wins}")

    if not args.no_runtime:
        run_runtime_demo(llm, retriever, profile, fewshot)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
