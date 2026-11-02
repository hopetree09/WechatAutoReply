"""LLM 评委：对已生成的 A/B 回复做盲评（GPU 空闲时单独跑）。

为什么要单独跑：A/B 生成阶段 3B 模型占着显存，本地 7B 评委塞不进同一张 8GB 卡。
所以流程是「先生成并落盘 → 进程退出释放显存 → 再跑评委」。

用法：
    python scripts/judge_replies.py --report data/eval/ab_report.json
    ... --judge-model qwen2.5:7b-instruct-q4_K_M --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import LlmSettings  # noqa: E402
from wagent.eval.ab import LLMJudge  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "data" / "eval" / "ab_report.json"))
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--out", default=str(ROOT / "data" / "eval" / "judge_report.md"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[!] 找不到 A/B 报告：{report_path}（先跑 scripts/eval_ab.py）")
        return 2
    data = json.loads(report_path.read_text(encoding="utf-8"))
    details = data.get("details", [])[: args.limit]
    if not details:
        print("[!] 报告里没有明细")
        return 2

    client = LLMClient(LlmSettings(provider="local", local_model=args.judge_model))
    judge = LLMJudge(client)

    base_wins = tuned_wins = ties = 0
    lines = [
        "# LLM 评委盲评结果",
        "",
        f"- 评委模型：`{args.judge_model}`（成对比较，交换 A/B 位置抵消偏差）",
        f"- 样本数：{len(details)}",
        "",
        "| # | 对方 | base 回复 | LoRA 回复 | 胜者 |",
        "|---|---|---|---|---|",
    ]
    for index, item in enumerate(details, start=1):
        reply_a = item["base"]["reply"]
        reply_b = item["tuned"]["reply"]
        context = item.get("context_text", "")
        verdict = judge.compare(context, reply_a, reply_b)
        if verdict.winner == "A":
            base_wins += 1
        elif verdict.winner == "B":
            tuned_wins += 1
        else:
            ties += 1
        lines.append(
            f"| {index} | {item.get('latest') or '（见上下文）'} | {reply_a} | {reply_b} | {verdict.winner} |"
        )
        print(f"  #{index} base={reply_a!r} tuned={reply_b!r} → {verdict.winner} {verdict.reason[:40]}")

    decided = base_wins + tuned_wins
    win_rate = round(tuned_wins / decided, 3) if decided else 0.0
    lines += [
        "",
        "## 汇总",
        "",
        f"- base 胜：{base_wins}",
        f"- LoRA 胜：{tuned_wins}",
        f"- 平局：{ties}",
        f"- LoRA 胜率（不含平局）：{win_rate}",
        "",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[judge] base={base_wins} tuned={tuned_wins} tie={ties} LoRA 胜率={win_rate}")
    print(f"[judge] 报告：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
