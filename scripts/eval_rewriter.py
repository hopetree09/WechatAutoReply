"""评测 rewriter：held-out 30 条真实样本上，「7B 草稿」vs「3B rewriter 改写」。

指标：
- style_similarity：改写结果与「我当时真实回复」的字级 2-gram 重叠（越高越像本人）
- 草稿保真度：改写结果与草稿的 2-gram 重叠（越高越说明没丢内容）
- AI 味 / 长度 / 综合分

生成阶段（会加载 3B，约 2 分钟）：
    python scripts/eval_rewriter.py
盲评阶段（3B 进程退出后跑 7B 评委）：
    python scripts/judge_rewriter.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.agent.two_stage import build_rewrite_messages, build_rewrite_simple  # noqa: E402
from wagent.eval.metrics import jaccard, score_reply  # noqa: E402
from wagent.llm.prompt import clean_reply  # noqa: E402
from wagent.memory import MemoryStore  # noqa: E402
from wagent.perception.types import Box, Message, Speaker  # noqa: E402
from wagent.train.dataset import build_samples, split_samples  # noqa: E402

DATA = ROOT / "data" / "train_rewriter"
OUT = ROOT / "data" / "eval" / "rewriter"


def _msg(record) -> Message:
    return Message(
        speaker=Speaker.ME if record.speaker == "me" else Speaker.OTHER,
        text=record.text,
        box=Box(0, 0, 1, 1),
    )


def fidelity(text: str, draft: str) -> float:
    """改写结果与草稿的 2-gram 重叠（衡量有没有丢内容）。"""
    return jaccard(text, draft, n=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="rewriter held-out 评测")
    parser.add_argument("--person", default="张伟")
    parser.add_argument("--db", default=str(ROOT / "data" / "agent.db"))
    parser.add_argument(
        "--base",
        default=str(ROOT / "models" / "models" / "Qwen--Qwen2.5-3B-Instruct" / "snapshots" / "master"),
    )
    parser.add_argument("--adapter", default=str(ROOT / "data" / "lora" / "qwen25-3b-rewriter"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--simple", action="store_true", help="用极简改写 prompt（rewriter v2）")
    args = parser.parse_args()

    cases = json.loads((DATA / "eval_cases.json").read_text(encoding="utf-8"))
    store = MemoryStore(Path(args.db))
    records = store.recent_messages(args.person, limit=200000)
    samples = build_samples(records, window=6, min_reply_chars=2, max_reply_chars=60)
    _train, val = split_samples(samples, val_ratio=0.1, seed=args.seed)
    import random

    random.Random(args.seed).shuffle(val)
    heldout = val[: len(cases)]
    references = [sample.text for sample in store.style_samples(limit=100)]
    store.close()

    from wagent.train.lora import LoraChatModel

    rewriter = LoraChatModel(base_model=args.base, adapter_dir=args.adapter, max_new_tokens=48)
    rows = []
    for case, sample in zip(cases, heldout, strict=False):
        context = [_msg(record) for record in sample.context]
        turns = (
            build_rewrite_simple(sample.latest, case["draft"])
            if args.simple
            else build_rewrite_messages(context, sample.latest, case["draft"], fewshot=True)
        )
        rewritten = clean_reply(rewriter.chat(turns).text)
        draft_score = score_reply(case["draft"], references=references)
        rewrite_score = score_reply(rewritten, references=references)
        rows.append(
            {
                "latest": sample.latest,
                "draft": case["draft"],
                "rewrite": rewritten,
                "reference": sample.reply,
                "draft_score": draft_score.as_dict(),
                "rewrite_score": rewrite_score.as_dict(),
                "fidelity": round(fidelity(rewritten, case["draft"]), 3),
            }
        )
        print(
            f"[{len(rows):02d}] 对方: {sample.latest[:18]}\n"
            f"     草稿: {case['draft']}\n"
            f"     改写: {rewritten}  (fidelity={rows[-1]['fidelity']:.2f})\n"
            f"     真实: {sample.reply}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    print("\n" + "=" * 60)
    for label, score_key, text_key in (
        ("草稿", "draft_score", "draft"),
        ("改写", "rewrite_score", "rewrite"),
    ):
        print(
            f"{label:<4} 综合 {sum(r[score_key]['score'] for r in rows) / n:.3f} | "
            f"风格 {sum(r[score_key]['style'] for r in rows) / n:.3f} | "
            f"AI 味 {sum(r[score_key]['ai_flavor'] for r in rows) / n:.3f} | "
            f"长度 {sum(len(r[text_key]) for r in rows) / n:.1f} 字"
        )
    print(f"平均草稿保真度: {sum(r['fidelity'] for r in rows) / n:.2f}")
    print(f"[eval] 结果写入 {OUT / 'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
