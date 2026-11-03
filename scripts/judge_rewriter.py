"""rewriter held-out 盲评：7B 评委比较「草稿」vs「rewriter 改写」。

前置：先跑 ``python scripts/eval_rewriter.py``（生成阶段，进程退出释放显存）。
用法：``python scripts/judge_rewriter.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import get_settings  # noqa: E402
from wagent.eval.ab import LLMJudge  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402

DATA = ROOT / "data" / "eval" / "rewriter" / "eval.json"


def main() -> int:
    rows = json.loads(DATA.read_text(encoding="utf-8"))["rows"]
    settings = get_settings()
    judge = LLMJudge(LLMClient(settings.llm))
    wins = {"draft": 0, "rewrite": 0, "tie": 0}
    for row in rows:
        result = judge.compare(f"对方刚发来：{row['latest']}", row["draft"], row["rewrite"])
        key = {"A": "draft", "B": "rewrite"}.get(result.winner, "tie")
        wins[key] += 1
    total = len(rows)
    print(f"[judge] 草稿 {wins['draft']} / rewriter {wins['rewrite']} / 平 {wins['tie']}")
    print(f"[judge] rewriter 盲评胜率 {wins['rewrite'] / total:.1%}（n={total}）")
    report = ROOT / "data" / "eval" / "rewriter" / "judge_report.md"
    report.write_text(
        "# rewriter held-out 盲评\n\n"
        + "\n".join(
            f"- {row['latest'][:20]} | draft={row['draft']} | rewrite={row['rewrite']}" for row in rows
        )
        + f"\n\n结果：草稿 {wins['draft']} / rewriter {wins['rewrite']} / 平 {wins['tie']}\n",
        encoding="utf-8",
    )
    print(f"[judge] 报告写入 {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
