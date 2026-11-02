"""导出「与推理时 prompt 一致」的 LoRA 训练集。

为什么需要它：之前训练集的 system 是一句短提示（DEFAULT_SYSTEM），而推理时
`build_messages()` 会塞「自述画像 + 层级记忆 + few-shot」的长 system prompt。
3B 模型在 train/inference prompt 分布不一致时会退化（复读口头禅、胡言乱语）。
这个脚本把**每条训练样本的 system 都按推理时的真实 prompt 构造**（含画像与记忆块），
训练出来的 adapter 才适配「三件套」。

用法：
    python scripts/export_matched_training_data.py --person 吴浩毅 --limit 2000
    python scripts/train_lora.py --data data/train_real_matched --out data/lora/qwen25-3b-real-matched --limit-samples 800
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.agent.intent import needs_memory  # noqa: E402
from wagent.config import get_settings  # noqa: E402
from wagent.llm.prompt import build_messages, load_fewshot, load_profile  # noqa: E402
from wagent.memory import MemoryQuery, MemoryRetriever, MemoryStore, VectorIndex  # noqa: E402
from wagent.memory.embeddings import OllamaEmbedder  # noqa: E402
from wagent.perception.types import Box, Message, Speaker  # noqa: E402
from wagent.train.dataset import build_samples, split_samples, write_jsonl  # noqa: E402


def to_inference_messages(
    sample,
    *,
    retriever: MemoryRetriever,
    person: str,
    profile: str,
    fewshot: list[str],
) -> dict:
    """按推理时的方式构造 system+user，assistant 用真实回复。"""
    context = [
        Message(
            speaker=Speaker.ME if record.speaker == "me" else Speaker.OTHER,
            text=record.text,
            box=Box(0, 0, 1, 1),
        )
        for record in sample.context
    ]
    latest = sample.latest
    memory = ""
    if needs_memory(latest):
        query = MemoryQuery(person=person, latest_text=latest, now=datetime.now().astimezone())
        memory = retriever.build(query).to_prompt_block()
    turns = build_messages(
        context, latest, profile=profile, fewshot=fewshot, limit=12, memory_block=memory
    )
    return {
        "messages": [
            {"role": "system", "content": turns[0].content},
            {"role": "user", "content": turns[1].content},
            {"role": "assistant", "content": sample.reply},
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="导出与推理 prompt 一致的训练集")
    parser.add_argument("--person", required=True)
    parser.add_argument("--db", default=str(ROOT / "data" / "agent.db"))
    parser.add_argument("--out", default=str(ROOT / "data" / "train_real_matched"))
    parser.add_argument("--limit", type=int, default=2000, help="训练集最多条数")
    parser.add_argument("--val-limit", type=int, default=200)
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
        print(f"[matched] 向量检索不可用，退回关键词：{exc}")

    records = store.recent_messages(args.person, limit=200000)
    samples = build_samples(records, window=6, min_reply_chars=2, max_reply_chars=60)
    train, val = split_samples(samples, val_ratio=0.1, seed=42)
    train = train[: args.limit]
    val = val[: args.val_limit]
    print(f"[matched] 全量 {len(samples)} 对；导出 train={len(train)} / val={len(val)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        payload = [
            to_inference_messages(
                sample, retriever=retriever, person=args.person, profile=profile, fewshot=fewshot
            )
            for sample in rows
        ]
        write_jsonl(payload, out / f"{split}.jsonl")
        memory_hit = sum(1 for row in payload if "【我记得的事】" in row["messages"][0]["content"])
        print(f"[matched] {split}: {len(payload)} 条，带记忆块 {memory_hit} 条 → {out / (split + '.jsonl')}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
