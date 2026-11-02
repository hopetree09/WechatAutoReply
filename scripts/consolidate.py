"""记忆巩固 CLI：把已导入的聊天记录抽成结构化事实 + 事件摘要。

用法：
    python scripts/consolidate.py --person 张三
    python scripts/consolidate.py --person 张三 --limit 300 --model qwen2.5:7b-instruct-q4_K_M
    python scripts/consolidate.py --all          # 对所有联系人跑一遍

输出：新增事实（带证据原文）、事件摘要、库统计。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import LlmSettings, get_settings  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402
from wagent.memory.consolidate import Consolidator, FactExtractor  # noqa: E402
from wagent.memory.store import MemoryStore  # noqa: E402
from wagent.memory.types import SELF_PERSON  # noqa: E402


def people_in_store(store: MemoryStore) -> list[str]:
    rows = store.conn.execute("SELECT DISTINCT person FROM messages ORDER BY person").fetchall()
    return [row["person"] for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--db", default="")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--provider", default="local", choices=["local", "cloud"])
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(args.db) if args.db else settings.paths.db_path
    if not db_path.exists():
        print(f"[!] 数据库不存在：{db_path}（先 python -m wagent.import_chat）")
        return 2

    store = MemoryStore(db_path)
    llm = LLMClient(LlmSettings(provider=args.provider, local_model=args.model))
    consolidator = Consolidator(
        store, FactExtractor(llm), summarize=not args.no_summary
    )

    people = people_in_store(store) if args.all else [args.person]
    people = [p for p in people if p]
    if not people:
        print("[!] 没有可巩固的联系人（--person 或 --all）")
        return 2

    for person in people:
        result = consolidator.consolidate_person(person, limit=args.limit)
        print(f"[{person}] {result.describe()}")
        for fact in store.active_facts(person):
            print(f"  [对方] {fact.subject} {fact.predicate}: {fact.value} "
                  f"(conf={fact.confidence}, 证据={fact.evidence_text[:20]})")
        for fact in store.active_facts(SELF_PERSON):
            print(f"  [我  ] {fact.subject} {fact.predicate}: {fact.value}")
        for episode in store.recent_episodes(person, limit=3):
            print(f"  [事件] {episode.ts[:10]} {episode.summary}")

    print("[db]", store.stats())
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
