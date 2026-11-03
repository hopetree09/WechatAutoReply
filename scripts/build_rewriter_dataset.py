"""构造 rewriter 训练集：7B 生成草稿 → 3B 学习「把草稿改写成我的真实回复」。

训练对格式（与推理时的改写 prompt 完全一致）：
    system: REWRITE_SYSTEM
    user  : 最近聊天记录 + 对方刚发来 + 改写示例 + 草稿
    assistant: 我当时真实回复

同时导出一份 held-out 评测集（draft + 真实回复），用于训完后对比「草稿 vs rewriter」。

用法：
    python scripts/build_rewriter_dataset.py --person 张伟 --limit 500 --eval-cases 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.agent.intent import needs_memory  # noqa: E402
from wagent.agent.two_stage import REWRITE_SYSTEM, build_rewrite_messages  # noqa: E402
from wagent.config import get_settings  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402
from wagent.llm.prompt import build_messages, clean_reply, load_fewshot, load_profile  # noqa: E402
from wagent.memory import MemoryQuery, MemoryRetriever, MemoryStore, VectorIndex  # noqa: E402
from wagent.memory.embeddings import OllamaEmbedder  # noqa: E402
from wagent.perception.types import Box, Message, Speaker  # noqa: E402
from wagent.train.dataset import build_samples, split_samples, write_jsonl  # noqa: E402

OUT = ROOT / "data" / "train_rewriter"


def to_messages(sample) -> list[Message]:
    return [
        Message(
            speaker=Speaker.ME if record.speaker == "me" else Speaker.OTHER,
            text=record.text,
            box=Box(0, 0, 1, 1),
        )
        for record in sample.context
    ]


def make_draft(
    llm: LLMClient,
    retriever: MemoryRetriever,
    person: str,
    sample,
    *,
    profile: str,
    fewshot: list[str],
) -> tuple[str, bool]:
    latest = sample.latest
    memory = ""
    if needs_memory(latest):
        query = MemoryQuery(person=person, latest_text=latest, now=datetime.now().astimezone())
        memory = retriever.build(query).to_prompt_block()
    turns = build_messages(
        to_messages(sample), latest, profile=profile, fewshot=fewshot, memory_block=memory
    )
    draft = clean_reply(llm.chat(turns).text)
    return draft, bool(memory.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="构造 rewriter 训练集（7B 草稿 → 我的回复）")
    parser.add_argument("--person", required=True)
    parser.add_argument("--db", default=str(ROOT / "data" / "agent.db"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--eval-cases", type=int, default=30)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = get_settings()
    profile = load_profile(settings.paths.profile_path)
    fewshot = load_fewshot(settings.paths.fewshot_dir)
    store = MemoryStore(Path(args.db))
    retriever = MemoryRetriever(store)
    try:
        embedder = OllamaEmbedder()
        vector_index = VectorIndex(store)
        vector_index.sync(embedder)
        retriever = MemoryRetriever(store, vector_index=vector_index, embedder=embedder)
    except Exception as exc:
        print(f"[rewriter] 向量检索不可用，退回关键词：{exc}")

    records = store.recent_messages(args.person, limit=200000)
    samples = build_samples(records, window=6, min_reply_chars=2, max_reply_chars=60)
    train, val = split_samples(samples, val_ratio=0.1, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)
    train = train[: args.limit]
    eval_cases = val[: args.eval_cases]
    print(f"[rewriter] 训练样本 {len(train)}，评测样本 {len(eval_cases)}（7B 逐条生成草稿…）")

    llm = LLMClient(settings.llm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def rows_for(items: list, *, with_reference: bool) -> list[dict]:
        rows: list[dict] = []
        started = time.time()
        for index, sample in enumerate(items, start=1):
            draft, memory_hit = make_draft(
                llm, retriever, args.person, sample, profile=profile, fewshot=fewshot
            )
            if not draft:
                continue
            context = to_messages(sample)
            rewrite_turns = build_rewrite_messages(context, sample.latest, draft, fewshot=True)
            row = {
                "messages": [
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": rewrite_turns[1].content},
                ]
            }
            if with_reference:
                row["messages"].append({"role": "assistant", "content": sample.reply})
            rows.append({**row, "_latest": sample.latest, "_draft": draft, "_memory": memory_hit})
            if index % 50 == 0:
                elapsed = time.time() - started
                print(f"  已生成 {index}/{len(items)}（{elapsed:.0f}s，约 {elapsed / index:.2f}s/条）")
        return rows

    train_rows = rows_for(train, with_reference=True)
    eval_rows = rows_for(eval_cases, with_reference=False)

    write_jsonl(train_rows, out / "train.jsonl")
    (out / "eval_cases.json").write_text(
        json.dumps(
            [
                {
                    "latest": row["_latest"],
                    "draft": row["_draft"],
                    "reference": reference,
                    "memory_hit": row["_memory"],
                }
                for row, reference in zip(eval_rows, [s.reply for s in eval_cases], strict=False)
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lengths = [len(row["_draft"]) for row in train_rows]
    print(
        f"[rewriter] 写出 {out/'train.jsonl'}（{len(train_rows)} 条，草稿均长 "
        f"{sum(lengths) / len(lengths):.1f} 字）；评测集 {out/'eval_cases.json'}（{len(eval_rows)} 条）"
    )
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
