"""把 rewriter v1 训练集改写成「极简改写 prompt」版本（草稿紧贴指令、无长上下文）。

v1 的问题：改写 prompt 里塞了上下文 + 3 个改写示例，3B 的注意力被上下文/示例带跑，
完全无视草稿（训练集上的 fidelity 也是 0）。这里从已生成的 train.jsonl 里抽出
latest / draft，用 ``build_rewrite_simple`` 重建训练对，避免重新用 7B 生成草稿。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.agent.two_stage import REWRITE_SIMPLE_SYSTEM, build_rewrite_simple  # noqa: E402

SRC = ROOT / "data" / "train_rewriter" / "train.jsonl"
OUT = ROOT / "data" / "train_rewriter_simple" / "train.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows = []
    for row in rows:
        messages = row["messages"]
        content = messages[1]["content"]
        latest = ""
        drafts = re.findall(r"对方刚发来：([^\n]+)", content)
        if drafts:
            latest = drafts[0].strip()
        elif "对方刚发来：" in content:
            latest = content.split("对方刚发来：", 1)[1].split("\n", 1)[0].strip()
        draft_candidates = re.findall(r"草稿：([^\n]+)", content)
        if not draft_candidates or not latest:
            continue
        draft = draft_candidates[-1].strip()
        turns = build_rewrite_simple(latest, draft)
        out_rows.append(
            {
                "messages": [
                    {"role": "system", "content": REWRITE_SIMPLE_SYSTEM},
                    {"role": "user", "content": turns[1].content},
                    {"role": "assistant", "content": messages[2]["content"]},
                ]
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    print(f"[rebuild] {len(rows)} 行 → {len(out_rows)} 行，写入 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
