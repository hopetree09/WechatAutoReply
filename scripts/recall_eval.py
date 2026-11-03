"""记忆召回评测：不依赖生成模型，直接测「该记起来的东西有没有被检索到」。

做法：
1. 从记忆库取真实事实卡（带证据），每条构造一个「回忆式问题」；
2. 用项目的 :class:`MemoryRetriever` 检索（向量 + 关键词 + 时效融合）；
3. 看目标事实是否出现在 top-1 / top-3 的 facts block 里，算 recall@1 / recall@3 / MRR。

这比「盲评回复像不像」更能体现层级记忆本身的价值。

用法：
    python scripts/recall_eval.py --db data/agent.db --person 张伟 --limit 30
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.memory import MemoryQuery, MemoryRetriever, MemoryStore  # noqa: E402

try:  # 向量检索可选：bge-m3 不在就退回关键词/时效
    from wagent.memory import OllamaEmbedder, VectorIndex
except ImportError:  # pragma: no cover
    OllamaEmbedder = None
    VectorIndex = None

QUESTION_TEMPLATES = (
    "你还记得我之前说过「{value}」吗？",
    "我们之前聊过的那个「{value}」，你还记得吗？",
    "帮我回忆一下「{value}」这件事",
)


def build_question(value: str, index: int) -> str:
    template = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]
    return template.format(value=value[:40])


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def evaluate(store: MemoryStore, person: str, limit: int, seed: int = 42) -> dict:
    facts = [fact for fact in store.active_facts(person, limit=400) if fact.value.strip()]
    facts += [fact for fact in store.active_facts("__self__", limit=400) if fact.value.strip()]
    if not facts:
        raise SystemExit("[!] 记忆库里没有事实卡，先跑 consolidate.py")
    rng = random.Random(seed)
    rng.shuffle(facts)
    facts = facts[:limit]

    vector_index = embedder = None
    try:
        embedder = OllamaEmbedder()
        vector_index = VectorIndex(store)
        vector_index.sync(embedder)
    except Exception as exc:
        print(f"[recall] 向量检索不可用，退回关键词：{exc}")

    retriever = MemoryRetriever(store, vector_index=vector_index, embedder=embedder)
    now = datetime.now().astimezone()
    hits1 = hits3 = 0
    reciprocal_rank = 0.0
    details: list[dict] = []
    for index, fact in enumerate(facts):
        question = build_question(fact.value, index)
        context = retriever.build(
            MemoryQuery(person=person, latest_text=question, now=now, max_facts=8)
        )
        needle = _normalize(fact.value)[:24]
        rank = 0
        lines = [line for line in context.facts_block.splitlines() if line.strip()]
        for position, line in enumerate(lines, start=1):
            if needle and needle in _normalize(line):
                rank = position
                break
        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 3:
            hits3 += 1
        if rank:
            reciprocal_rank += 1.0 / rank
        details.append(
            {
                "fact": f"{fact.subject}/{fact.predicate}",
                "subject": fact.subject,
                "question": question,
                "rank": rank,
                "n_lines": len(lines),
                "vector": vector_index is not None,
            }
        )

    n = len(facts)
    return {
        "n": n,
        "recall@1": hits1 / n if n else 0.0,
        "recall@3": hits3 / n if n else 0.0,
        "mrr": reciprocal_rank / n if n else 0.0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="记忆召回评测（recall@1 / recall@3 / MRR）")
    parser.add_argument("--db", default=str(ROOT / "data" / "agent.db"))
    parser.add_argument("--person", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    store = MemoryStore(Path(args.db))
    try:
        result = evaluate(store, args.person, args.limit)
    finally:
        store.close()

    print(f"[recall] 事实 {result['n']} 条："
          f"recall@1={result['recall@1']:.2f}, "
          f"recall@3={result['recall@3']:.2f}, MRR={result['mrr']:.2f}")
    for item in result["details"][:10]:
        print(f"  rank={item['rank']:<2} {item['fact']:<28} q={item['question'][:40]}")
    if args.out:
        import json

        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[recall] 结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
