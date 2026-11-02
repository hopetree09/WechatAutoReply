"""A/B 评测：基座模型 vs LoRA 模型。

评测集用训练时切出来的 val.jsonl（每行包含 user 上下文与 assistant 参考回复），
同一份 prompt 分别用「关掉 adapter」和「挂上 adapter」生成，然后：
1. 规则指标：风格相似度 / AI 味 / 长度自然度 / 复读惩罚；
2. LLM 评委：本地 7B 成对比较，交换位置抵消偏差；
3. 输出 markdown + json 报告。

用法：
    python scripts/eval_ab.py --base models/Qwen2.5-3B-Instruct \
        --adapter data/lora/qwen25-3b-style --data data/train/val.jsonl \
        --limit 30 --judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import LlmSettings  # noqa: E402
from wagent.eval.ab import ABPrompt, LLMJudge, run_ab  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402
from wagent.train.lora import generate_reply, load_for_inference  # noqa: E402


def load_prompts(path: Path, limit: int) -> tuple[list[ABPrompt], list[str]]:
    prompts: list[ABPrompt] = []
    references: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages", [])
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        if not user or not assistant:
            continue
        latest = ""
        for marker in ("对方刚发来：", "对方刚发来:"):
            if marker in user:
                latest = user.split(marker, 1)[1].split("\n", 1)[0].strip()
                break
        prompts.append(ABPrompt(context_text=user, latest=latest, reference=assistant))
        references.append(assistant)
        if len(prompts) >= limit:
            break
    return prompts, references[:200]


def write_report(result, out_dir: Path, *, judge_used: bool, adapter: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_report.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# LoRA A/B 评测报告",
        "",
        f"- 样本数：{result.n}",
        f"- adapter：`{adapter}`",
        f"- 评委：{'本地 LLM 成对比较（交换位置）' if judge_used else '未启用（仅规则指标）'}",
        "",
        "## 规则指标（越高越好；AI 味越低越好）",
        "",
        "| 模型 | 综合分 | 风格相似 | AI 味 | 长度自然 |",
        "|---|---|---|---|---|",
        f"| base | {result.base_metrics.get('score', 0)} | {result.base_metrics.get('style', 0)} "
        f"| {result.base_metrics.get('ai_flavor', 0)} | {result.base_metrics.get('length', 0)} |",
        f"| tuned | {result.tuned_metrics.get('score', 0)} | {result.tuned_metrics.get('style', 0)} "
        f"| {result.tuned_metrics.get('ai_flavor', 0)} | {result.tuned_metrics.get('length', 0)} |",
        "",
        "## 评委胜负",
        "",
        f"- base 胜：{result.a_wins}",
        f"- tuned 胜：{result.b_wins}",
        f"- 平局：{result.ties}",
        f"- tuned 胜率（不含平局）：{result.win_rate('b')}",
        "",
    ]
    for detail in result.details[:10]:
        lines.append(f"- 对方：{detail['latest'] or '（见上下文）'}")
        lines.append(f"  - base：{detail['base']['reply']}")
        lines.append(f"  - tuned：{detail['tuned']['reply']} → {detail['winner']}")
    path = out_dir / "ab_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-3B-Instruct"))
    parser.add_argument("--adapter", default=str(ROOT / "data" / "lora" / "qwen25-3b-style"))
    parser.add_argument("--data", default=str(ROOT / "data" / "train" / "val.jsonl"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--judge", action="store_true", help="用本地 7B 做评委")
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--out", default=str(ROOT / "data" / "eval"))
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[!] 评测集不存在：{data_path}（先跑 scripts/train_lora.py --synth）")
        return 2
    prompts, references = load_prompts(data_path, args.limit)
    print(f"[eval] 样本 {len(prompts)}，adapter={args.adapter}")

    model, tokenizer = load_for_inference(
        args.base, args.adapter, load_in_4bit=not args.no_4bit
    )

    def gen_base(prompt: ABPrompt) -> str:
        with model.disable_adapter():
            return generate_reply(model, tokenizer, prompt.context_text)

    def gen_tuned(prompt: ABPrompt) -> str:
        return generate_reply(model, tokenizer, prompt.context_text)

    judge = None
    if args.judge:
        settings = LlmSettings(provider="local", local_model=args.judge_model)
        judge = LLMJudge(LLMClient(settings))

    result = run_ab(prompts, gen_base, gen_tuned, judge=judge, references=references)
    report_path = write_report(result, Path(args.out), judge_used=judge is not None, adapter=args.adapter)

    print("[eval] base  :", result.base_metrics)
    print("[eval] tuned :", result.tuned_metrics)
    print(f"[eval] 评委：base={result.a_wins} tuned={result.b_wins} tie={result.ties} "
          f"tuned_win_rate={result.win_rate('b')}")
    print(f"[eval] 报告：{report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
