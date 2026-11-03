"""从本地记忆库导出 LoRA 训练集（ChatML JSONL）。

用法：
    python scripts\\export_training_data.py --out data/train
    ... --person 张三 --window 6 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import get_settings  # noqa: E402
from wagent.memory.store import MemoryStore  # noqa: E402
from wagent.train.dataset import export_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "train"))
    parser.add_argument("--db", default="")
    parser.add_argument("--person", default=None)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(args.db) if args.db else settings.paths.db_path
    if not db_path.exists():
        print(f"[!] 数据库不存在：{db_path}（先运行 python -m wagent.import_chat）")
        return
    store = MemoryStore(db_path)
    try:
        stats = export_dataset(
            store,
            Path(args.out),
            person=args.person,
            window=args.window,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    finally:
        store.close()

    print(f"[dataset] 联系人：{', '.join(stats['people']) or '（无）'}")
    print(
        f"[dataset] 样本 {stats['samples']} 条 → train {stats['train']} / val {stats['val']}；"
        f"回复平均 {stats['avg_reply_chars']} 字，最长 {stats['max_reply_chars']} 字"
    )
    print(f"[dataset] 已写入 {Path(args.out)}/train.jsonl、val.jsonl")


if __name__ == "__main__":
    main()
