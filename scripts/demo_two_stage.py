"""两段式（7B 草稿 → 3B LoRA 改写）的 8 场景对比。

阶段一（默认，会加载 3B 模型）：
    python scripts/demo_two_stage.py --stage gen
阶段二（3B 进程退出后再跑，7B 只用来盲评，避免显存争用）：
    python scripts/demo_two_stage.py --stage judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_compare import CASES, _msg, generate_short, memory_block  # noqa: E402
from wagent.agent.intent import needs_memory  # noqa: E402
from wagent.agent.state import Trigger  # noqa: E402
from wagent.agent.two_stage import build_rewrite_messages  # noqa: E402
from wagent.config import get_settings  # noqa: E402
from wagent.eval.metrics import score_reply  # noqa: E402
from wagent.llm.client import LLMClient  # noqa: E402
from wagent.llm.prompt import build_messages, clean_reply, load_fewshot, load_profile  # noqa: E402
from wagent.memory import MemoryRetriever, MemoryStore, VectorIndex  # noqa: E402
from wagent.memory.embeddings import OllamaEmbedder  # noqa: E402
from wagent.perception.types import Box, Message, Speaker  # noqa: E402

SESSION = "吴浩毅"
RESULTS = ROOT / "data" / "eval" / "two_stage" / "results.json"


def to_trigger(case) -> Trigger:
    latest = Message(speaker=Speaker.OTHER, text=case.latest, box=Box(0, 0, 1, 1))
    context = [_msg(speaker, text) for speaker, text in case.context]
    return Trigger(message=latest, context=context, session=SESSION)


def _unload_ollama(models: list[str]) -> None:
    """让 Ollama 立刻释放显存（keep_alive=0），给 transformers 模型腾地方。"""
    import urllib.request

    for model in models:
        try:
            body = json.dumps({"model": model, "keep_alive": 0}).encode()
            request = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=15).read()
            print(f"[two-stage] 已释放 Ollama 模型：{model}")
        except Exception as exc:
            print(f"[two-stage] 释放 {model} 失败（可忽略）：{exc}")


def make_retriever(store: MemoryStore) -> MemoryRetriever:
    retriever = MemoryRetriever(store)
    try:
        embedder = OllamaEmbedder()
        vector_index = VectorIndex(store)
        vector_index.sync(embedder)
        return MemoryRetriever(store, vector_index=vector_index, embedder=embedder)
    except Exception as exc:
        print(f"[two-stage] 向量检索不可用，退回关键词：{exc}")
        return retriever


def stage_gen(args: argparse.Namespace) -> int:
    if not RESULTS.parent.exists():
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    profile = load_profile(settings.paths.profile_path)
    fewshot = load_fewshot(settings.paths.fewshot_dir)
    store = MemoryStore(settings.paths.db_path)
    retriever = make_retriever(store)
    references = [sample.text for sample in store.style_samples(limit=100)]

    draft_llm = LLMClient(settings.llm)

    # 阶段 A：7B 出草稿（含画像/记忆），先全部生成完
    drafts: list[dict[str, str]] = []
    for case in CASES:
        trigger = to_trigger(case)
        memory = memory_block(retriever, case.latest) if needs_memory(case.latest) else ""
        turns = build_messages(
            trigger.context,
            trigger.message.text,
            profile=profile,
            fewshot=fewshot,
            memory_block=memory,
        )
        draft = clean_reply(draft_llm.chat(turns).text)
        drafts.append(
            {
                "topic": case.topic,
                "latest": case.latest,
                "draft": draft,
                "memory": memory,
            }
        )
        print(f"[gen] {case.topic}: {draft}")

    # 释放 7B/bge-m3，给 3B LoRA 腾显存（8GB 卡两个大模型会 OOM）
    _unload_ollama([settings.llm.local_model, "bge-m3:latest"])

    from wagent.train.lora import LoraChatModel

    rewrite_llm = LoraChatModel(
        base_model=args.base,
        adapter_dir=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )

    # 阶段 B：3B LoRA 改写（短 system prompt，贴着它的训练分布）
    rows: list[dict[str, object]] = []
    agg: dict[str, list[float]] = {key: [0.0, 0.0, 0.0] for key in ("draft", "final", "direct")}
    for case, item in zip(CASES, drafts, strict=True):
        rewrite_result = rewrite_llm.chat(
            build_rewrite_messages(
                [_msg(speaker, text) for speaker, text in case.context],
                case.latest,
                item["draft"],
            )
        )
        final = clean_reply(rewrite_result.text)
        direct = generate_short(rewrite_llm, case)  # 3B 直接风格回复（无 7B 草稿）
        row_scores: dict[str, dict[str, float]] = {}
        for key, text in (("draft", item["draft"]), ("final", final), ("direct", direct)):
            score = score_reply(text, references=references)
            row_scores[key] = score.as_dict()
            agg[key][0] += score.score
            agg[key][1] += score.ai_flavor
            agg[key][2] += len(text)
        rows.append(
            {
                "topic": case.topic,
                "latest": case.latest,
                "draft": item["draft"],
                "final": final,
                "direct": direct,
                "memory": item["memory"],
                "scores": row_scores,
            }
        )
        print(f"\n[{case.topic}] 对方: {case.latest}")
        print(f"  7B 草稿   : {item['draft']}")
        print(f"  两段式最终: {final}")
        print(f"  3B 直接    : {direct}")

    RESULTS.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[two-stage] 结果写入 {RESULTS}")
    count = len(rows)
    for key, (score_sum, flavor_sum, char_sum) in agg.items():
        print(
            f"{key:<8} 综合 {score_sum / count:.3f} | "
            f"AI 味 {flavor_sum / count:.3f} | 长度 {char_sum / count:.1f} 字"
        )
    store.close()
    return 0


def stage_rewrite(args: argparse.Namespace) -> int:
    """读已有草稿，用 few-shot 改写 prompt 重跑第二段（不需要 7B）。"""
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload["rows"]
    from wagent.train.lora import LoraChatModel

    rewriter = LoraChatModel(
        base_model=args.base,
        adapter_dir=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )
    for row in rows:
        case = next((c for c in CASES if c.topic == row["topic"]), None)
        if case is None:
            continue
        result = rewriter.chat(
            build_rewrite_messages(
                [_msg(speaker, text) for speaker, text in case.context],
                case.latest,
                row["draft"],
                fewshot=True,
            )
        )
        row["final_fs"] = clean_reply(result.text)
        score = score_reply(row["final_fs"])
        row["scores"]["final_fs"] = score.as_dict()
        print(f"[{row['topic']}] 草稿: {row['draft']}")
        print(f"  few-shot 改写: {row['final_fs']}  (len={len(row['final_fs'])}, score={score.score:.3f})")
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finals = [row["final_fs"] for row in rows]
    print(f"\n[rewrite] few-shot 改写平均长度 {sum(len(f) for f in finals) / len(finals):.1f} 字；结果已更新 {RESULTS}")
    return 0


def stage_judge(args: argparse.Namespace) -> int:
    from wagent.eval.ab import LLMJudge

    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload["rows"]
    settings = get_settings()
    judge = LLMJudge(LLMClient(settings.llm))
    wins = {"draft": 0, "final": 0, "tie": 0}
    print("[judge] 7B 草稿 vs 两段式最终（盲评）")
    for row in rows:
        result = judge.compare(
            f"对方刚发来：{row['latest']}",
            row["draft"],
            row["final"],
        )
        wins[{"A": "draft", "B": "final"}.get(result.winner, "tie")] += 1
        print(f"  [{row['topic']}] → {result.winner}：{result.reason[:48]}")
    total = len(rows)
    print(
        f"[judge] 草稿胜 {wins['draft']} / 两段式胜 {wins['final']} / 平 {wins['tie']}"
        f"（两段式胜率 {wins['final'] / total:.1%}）"
    )
    if any("final_fs" in row for row in rows):
        wins_fs = {"draft": 0, "final_fs": 0, "tie": 0}
        print("\n[judge] 7B 草稿 vs few-shot 改写（盲评）")
        for row in rows:
            if "final_fs" not in row:
                continue
            result = judge.compare(
                f"对方刚发来：{row['latest']}",
                row["draft"],
                row["final_fs"],
            )
            wins_fs[{"A": "draft", "B": "final_fs"}.get(result.winner, "tie")] += 1
            print(f"  [{row['topic']}] → {result.winner}：{result.reason[:48]}")
        print(
            f"[judge] 草稿胜 {wins_fs['draft']} / few-shot 胜 {wins_fs['final_fs']} / 平 {wins_fs['tie']}"
            f"（few-shot 胜率 {wins_fs['final_fs'] / total:.1%}）"
        )
    report = ROOT / "data" / "eval" / "two_stage" / "judge_report.md"
    report.write_text(
        "# 两段式 vs 7B 草稿 盲评\n\n"
        + "\n".join(f"- {row['topic']}: draft={row['draft']} | final={row['final']}" for row in rows)
        + f"\n\n结果：草稿 {wins['draft']} / 两段式 {wins['final']} / 平 {wins['tie']}\n",
        encoding="utf-8",
    )
    print(f"[judge] 报告写入 {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="两段式生成对比（7B 草稿 → 3B LoRA 改写）")
    parser.add_argument("--stage", choices=("gen", "judge", "rewrite"), default="gen")
    parser.add_argument(
        "--base",
        default=str(ROOT / "models" / "models" / "Qwen--Qwen2.5-3B-Instruct" / "snapshots" / "master"),
    )
    parser.add_argument("--adapter", default=str(ROOT / "data" / "lora" / "qwen25-3b-real"))
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    if args.stage == "gen":
        return stage_gen(args)
    if args.stage == "rewrite":
        return stage_rewrite(args)
    return stage_judge(args)


if __name__ == "__main__":
    sys.exit(main())
