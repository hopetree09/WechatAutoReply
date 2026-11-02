"""消融实验：有/无层级记忆（同一模型、同一批 prompt）。

用法：
    python scripts/ablation.py --limit 12
    python scripts/ablation.py --judge-model qwen2.5:7b-instruct-q4_K_M --out data/eval

流程：
1. 从 data/train_synth/val.jsonl 取评测对话；
2. A 组 prompt = 人设 + 风格规则 + few-shot；B 组额外拼上 L0-L5 记忆块；
3. 本地 7B 分别生成，规则指标打分 + LLM 盲评（交换 A/B 抵消位置偏差）；
4. 输出 data/eval/ablation_report.md / .json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import LlmSettings, get_settings  # noqa: E402
from wagent.eval.ab import ABPrompt, LLMJudge, run_ab  # noqa: E402
from wagent.eval.ablation import (  # noqa: E402
    has_memory,
    load_cases,
    memory_block_for,
    seed_memory_store,
)
from wagent.llm.client import LLMClient, MessageTurn  # noqa: E402
from wagent.llm.prompt import (  # noqa: E402
    build_system_prompt,
    clean_reply,
    load_fewshot,
    load_profile,
)
from wagent.memory.store import MemoryStore  # noqa: E402

PERSON = "张伟"


def build_case_prompts(cases, *, profile, fewshot, store, with_memory: bool, vector_index=None, embedder=None):
    """返回与 cases 等长的 [(system, user)] 列表。"""
    prompts = []
    for case in cases:
        memory_block = (
            memory_block_for(
                store, PERSON, case.latest, vector_index=vector_index, embedder=embedder
            )
            if with_memory
            else ""
        )
        system = build_system_prompt(profile, fewshot, memory_block=memory_block)
        prompts.append((system, case.user_content))
    return prompts


def make_generator(llm, prompts):
    def generate(prompt: ABPrompt) -> str:
        # run_ab 传入的 prompt.context_text 就是 case.user_content，用其反查对应 system
        for system, user in prompts:
            if user == prompt.context_text:
                result = llm.chat(
                    [MessageTurn("system", system), MessageTurn("user", user)],
                    temperature=0.8,
                    max_tokens=120,
                )
                return clean_reply(result.text)
        return ""

    return generate


def write_report(data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_report.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 消融实验：有 / 无层级记忆",
        "",
        f"- 评测样本：{data['n']}",
        f"- 模型：`{data['model']}`（同一模型，唯一变量是 prompt 是否注入记忆）",
        f"- 记忆命中率：{data['memory_hit_rate']}",
        "",
        "## 规则指标",
        "",
        "| 组 | 综合分 | 风格相似 | AI 味 | 长度自然 |",
        "|---|---|---|---|---|",
        f"| A 无记忆 | {data['base_metrics']['score']} | {data['base_metrics']['style']} "
        f"| {data['base_metrics']['ai_flavor']} | {data['base_metrics']['length']} |",
        f"| B 有记忆 | {data['memory_metrics']['score']} | {data['memory_metrics']['style']} "
        f"| {data['memory_metrics']['ai_flavor']} | {data['memory_metrics']['length']} |",
        "",
        "## LLM 盲评",
        "",
        f"- 无记忆胜：{data['base_wins']}",
        f"- 有记忆胜：{data['memory_wins']}",
        f"- 平局：{data['ties']}",
        f"- 有记忆胜率（不含平局）：{data['memory_win_rate']}",
        "",
        "## 样例",
        "",
    ]
    for item in data["details"][:8]:
        lines += [
            f"- 对方：{item['latest'] or '（见上下文）'}",
            f"  - 无记忆：{item['base']['reply']}",
            f"  - 有记忆：{item['memory']['reply']} → {item['winner']}",
        ]
    path = out_dir / "ablation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "train_synth" / "val.jsonl"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--out", default=str(ROOT / "data" / "eval"))
    args = parser.parse_args()

    val_path = Path(args.data)
    if not val_path.exists():
        print(f"[!] 评测集不存在：{val_path}（先 python scripts/train_lora.py --synth）")
        return 2

    cases = load_cases(val_path, limit=args.limit)
    if not cases:
        print("[!] 评测集为空")
        return 2

    settings = get_settings()
    profile = load_profile(settings.paths.profile_path)
    fewshot = load_fewshot(settings.paths.fewshot_dir)

    store = MemoryStore(":memory:")
    style_bank = [case.reference for case in cases] + [
        "行吧 先这样", "问题不大 我看看", "有啊 咋了", "可以 你定", "哈哈 也行",
    ]
    seed_memory_store(store, style_samples=style_bank)

    # 用本地 bge-m3 建向量索引，让 B 组走完整的混合检索
    vector_index = embedder = None
    try:
        from wagent.memory import OllamaEmbedder, VectorIndex

        embedder = OllamaEmbedder()
        vector_index = VectorIndex(store)
        print(f"[ablation] 向量索引：{vector_index.sync(embedder)}")
    except Exception as exc:
        print(f"[ablation] 向量检索不可用（退回关键词/recency）：{exc}")

    base_prompts = build_case_prompts(
        cases, profile=profile, fewshot=fewshot, store=store, with_memory=False
    )
    memory_prompts = build_case_prompts(
        cases,
        profile=profile,
        fewshot=fewshot,
        store=store,
        with_memory=True,
        vector_index=vector_index,
        embedder=embedder,
    )
    hit = sum(1 for system, _ in memory_prompts if has_memory(system))
    hit_rate = round(hit / len(cases), 3)

    llm = LLMClient(LlmSettings(provider="local", local_model=args.judge_model))
    print(f"[ablation] {len(cases)} 条样本，记忆命中率={hit_rate}，模型={args.judge_model}")
    print("[ablation] 生成 A（无记忆）...")
    gen_base = make_generator(llm, base_prompts)
    print("[ablation] 生成 B（有记忆）...")
    gen_memory = make_generator(llm, memory_prompts)

    judge = LLMJudge(LLMClient(LlmSettings(provider="local", local_model=args.judge_model)))
    references = list(style_bank)
    ab_prompts = [
        ABPrompt(context_text=case.user_content, latest=case.latest, reference=case.reference)
        for case in cases
    ]
    result = run_ab(ab_prompts, gen_base, gen_memory, judge=judge, references=references)

    data = {
        "n": result.n,
        "model": args.judge_model,
        "memory_hit_rate": hit_rate,
        "base_metrics": result.base_metrics,
        "memory_metrics": result.tuned_metrics,
        "base_wins": result.a_wins,
        "memory_wins": result.b_wins,
        "ties": result.ties,
        "memory_win_rate": result.win_rate("b"),
        "details": [
            {
                "latest": item["latest"],
                "base": {"reply": item["base"]["reply"]},
                "memory": {"reply": item["tuned"]["reply"]},
                "winner": item["winner"],
            }
            for item in result.details
        ],
    }
    report = write_report(data, Path(args.out))
    print("[ablation] 无记忆：", result.base_metrics)
    print("[ablation] 有记忆：", result.tuned_metrics)
    print(
        f"[ablation] 盲评：无记忆 {result.a_wins} / 有记忆 {result.b_wins} / 平 {result.ties} "
        f"（有记忆胜率 {result.win_rate('b')}）"
    )
    print(f"[ablation] 报告：{report}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
