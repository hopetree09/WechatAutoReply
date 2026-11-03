"""对比「裸模型 / +自动画像 / +画像+层级记忆」三组 prompt 的回复效果。

两种数据来源：
- 默认：从真实记忆库回放（取「对方说了一句 → 我下一句回了什么」的样本，可用参考回复打分）；
- ``--live``：直接截取当前微信窗口的最后一条对方消息，作为"刚发来"的消息做预览（不发送）。

用法：
    python scripts/reply_compare.py --person 张伟 --cases 5
    python scripts/reply_compare.py --live
    python scripts/reply_compare.py --person 张伟 --cases 5 --judge
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.agent.state import Trigger  # noqa: E402
from wagent.config import get_settings  # noqa: E402
from wagent.eval.metrics import score_reply  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402
from wagent.llm.prompt import build_messages, clean_reply, load_fewshot, load_profile  # noqa: E402
from wagent.memory import MemoryQuery, MemoryRetriever, MemoryStore, VectorIndex  # noqa: E402
from wagent.memory.embeddings import OllamaEmbedder  # noqa: E402
from wagent.perception.types import Box, Message, Speaker  # noqa: E402


@dataclass
class Case:
    session: str
    latest: str
    context: list[Message]
    reference: str = ""


def _to_message(speaker: str, text: str) -> Message:
    return Message(
        speaker=Speaker.ME if speaker == "me" else Speaker.OTHER,
        text=text,
        box=Box(0, 0, 1, 1),
    )


def build_cases_from_store(store: MemoryStore, person: str, *, cases: int, seed: int) -> list[Case]:
    """回放真实对话：找「对方消息 → 我下一条回复」的成对样本。"""
    records = store.recent_messages(person, limit=200000)
    candidates: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        if record.speaker != "other" or not record.text.strip():
            continue
        if index + 1 >= len(records):
            continue
        following = records[index + 1]
        if following.speaker != "me" or not following.text.strip():
            continue
        candidates.append((index, following.text.strip()))
    if not candidates:
        raise SystemExit("[!] 找不到「对方 → 我」的对话样本")
    rng = random.Random(seed)
    picked = rng.sample(candidates, min(cases, len(candidates)))
    out: list[Case] = []
    for index, reference in sorted(picked):
        latest = records[index]
        context = [
            _to_message(record.speaker, record.text)
            for record in records[max(0, index - 6) : index]
        ]
        out.append(Case(session=person, latest=latest.text.strip(), context=context, reference=reference))
    return out


def build_live_case() -> Case:
    """抓当前微信窗口的最后一条对方消息（真正的"刚收到"）。"""
    from wagent.im.wechat import WeChatAdapter
    from wagent.winapi import enable_dpi_awareness

    enable_dpi_awareness()
    settings = get_settings()
    adapter = WeChatAdapter(settings)
    frame = adapter.capture()
    messages = frame.messages
    last_other = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].speaker is Speaker.OTHER and messages[index].text.strip():
            last_other = index
            break
    if last_other is None:
        raise SystemExit("[!] 当前窗口里没有对方消息（先切到一个有对方发言的会话）")
    latest = messages[last_other]
    context = messages[max(0, last_other - 6) : last_other]
    return Case(session=frame.session or "unknown", latest=latest.text.strip(), context=context)


def memory_block_for(
    retriever: MemoryRetriever, case: Case, now: datetime
) -> tuple[str, MemoryQuery]:
    query = MemoryQuery(person=case.session, latest_text=case.latest, now=now)
    return retriever.build(query).to_prompt_block(), query


def make_trigger(case: Case) -> Trigger:
    latest = Message(speaker=Speaker.OTHER, text=case.latest, box=Box(0, 0, 1, 1))
    return Trigger(message=latest, context=list(case.context), session=case.session)


def run_case(llm, case: Case, *, profile: str, fewshot: list[str], memory: str) -> str:
    turns = build_messages(
        case.context,
        case.latest,
        profile=profile,
        fewshot=fewshot,
        limit=12,
        memory_block=memory,
    )
    result = llm.chat(turns)
    return clean_reply(result.text)


def main() -> int:
    parser = argparse.ArgumentParser(description="三组 prompt 的回复效果对比")
    parser.add_argument("--person", default="张伟")
    parser.add_argument("--db", default=str(ROOT / "data" / "agent.db"))
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true", help="用当前微信窗口的最后一条对方消息")
    parser.add_argument("--judge", action="store_true", help="额外跑 LLM 盲评（A vs C、B vs C）")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(settings.paths.profile_path)
    fewshot = load_fewshot(settings.paths.fewshot_dir)

    store = MemoryStore(Path(args.db))
    retriever = MemoryRetriever(store)
    embedder = vector_index = None
    try:
        embedder = OllamaEmbedder()
        vector_index = VectorIndex(store)
        vector_index.sync(embedder)
        retriever = MemoryRetriever(store, vector_index=vector_index, embedder=embedder)
    except Exception as exc:
        print(f"[compare] 向量检索不可用，退回关键词：{exc}")

    cases = [build_live_case()] if args.live else build_cases_from_store(
        store, args.person, cases=args.cases, seed=args.seed
    )
    references = [c.reference for c in cases if c.reference]
    references += [s.text for s in store.style_samples(limit=40)]

    llm = LLMClient(settings.llm)
    rows = []
    for case in cases:
        mem, _query = memory_block_for(retriever, case, datetime.now().astimezone())
        replies = {
            "A 裸模型": run_case(llm, case, profile="", fewshot=[], memory=""),
            "B +画像": run_case(llm, case, profile=profile, fewshot=fewshot, memory=""),
            "C +画像+记忆": run_case(llm, case, profile=profile, fewshot=fewshot, memory=mem),
        }
        scores = {
            key: score_reply(reply, references=references) if case.reference else None
            for key, reply in replies.items()
        }
        rows.append((case, replies, scores, bool(mem.strip())))
        print("\n" + "=" * 72)
        print(f"[{case.session}] 对方: {case.latest}")
        if case.reference:
            print(f"  参考(我当时的回复): {case.reference[:40]}")
        for key, reply in replies.items():
            print(f"  {key:<10}: {reply}")
        if case.reference:
            for key, score in scores.items():
                assert score is not None
                print(f"    {key:<10} 综合 {score.total:.3f} | 风格 {score.style:.3f} "
                      f"| AI味 {score.ai_flavor:.3f} | 长度 {len(replies[key])}")

    print("\n" + "=" * 72)
    for key in ("A 裸模型", "B +画像", "C +画像+记忆"):
        valid = []
        for _c, _r, _s, _m in rows:
            score = _s[key]
            if score is not None:
                valid.append(score)
        if not valid:
            continue
        print(f"{key:<12} 平均综合 {sum(s.total for s in valid) / len(valid):.3f} | "
              f"风格 {sum(s.style for s in valid) / len(valid):.3f} | "
              f"AI 味 {sum(s.ai_flavor for s in valid) / len(valid):.3f} | "
              f"平均长度 {sum(len(r[key]) for _c, r, _s, _m in rows) / len(rows):.1f} 字")
    print(f"[compare] 记忆命中：{sum(1 for *_x, m in rows if m)}/{len(rows)} 条样本带记忆块")

    if args.judge:
        from wagent.eval.ab import LLMJudge

        judge = LLMJudge(llm)
        wins = {"A 裸模型": 0, "B +画像": 0, "C +画像+记忆": 0, "平": 0}
        for case, _replies, _scores, _m in rows:
            if not case.reference:
                continue
            result = judge.compare(
                context_text="\n".join(f"{'我' if m.speaker is Speaker.ME else '对方'}: {m.text}" for m in case.context),
                reference=case.reference,
                reply_a=_replies["A 裸模型"],
                reply_b=_replies["C +画像+记忆"],
            )
            winner = "平" if result.winner == "tie" else ("C +画像+记忆" if result.winner == "b" else "A 裸模型")
            wins[winner] += 1
            print(f"[judge] {case.latest[:16]}… → {winner}（{result.reason[:40]}）")
        print(f"[judge] A vs C：{wins}")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
