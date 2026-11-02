"""训练 3B LoRA 风格模型。

用法（真实数据）：
    python scripts/train_lora.py --base models/Qwen/Qwen2.5-3B-Instruct \\
        --data data/train --out data/lora/qwen25-3b-style

没有真实数据时先用合成数据跑通链路：
    python scripts/train_lora.py --synth --epochs 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.train.dataset import build_samples, split_samples, write_jsonl  # noqa: E402
from wagent.train.lora import LoraConfig, train_lora  # noqa: E402
from wagent.train.synth import SynthConfig, generate_records  # noqa: E402


def build_synth_dataset(out_dir: Path, *, exchanges: int, seed: int) -> tuple[Path, Path]:
    records = generate_records(SynthConfig(exchanges=exchanges, seed=seed))
    samples = build_samples(records, window=6)
    train, val = split_samples(samples, val_ratio=0.1, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl([s.to_chatml() for s in train], out_dir / "train.jsonl")
    write_jsonl([s.to_chatml() for s in val], out_dir / "val.jsonl")
    print(f"[synth] 生成 {len(records)} 条消息 → 样本 {len(samples)}（train {len(train)} / val {len(val)}）")
    return out_dir / "train.jsonl", out_dir / "val.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-3B-Instruct"))
    parser.add_argument("--data", default=str(ROOT / "data" / "train"))
    parser.add_argument("--out", default=str(ROOT / "data" / "lora" / "qwen25-3b-style"))
    parser.add_argument("--synth", action="store_true", help="没有真实数据时用合成数据")
    parser.add_argument("--synth-exchanges", type=int, default=320)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true", help="关闭 4bit 量化（显存不足时反而更慢）")
    parser.add_argument("--limit-samples", type=int, default=0, help="只取前 N 条样本（冒烟测试）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    if args.synth or not train_file.exists():
        print("[data] 未找到真实训练集，使用合成数据")
        train_file, val_file = build_synth_dataset(
            data_dir, exchanges=args.synth_exchanges, seed=args.seed
        )

    config = LoraConfig(
        base_model=args.base,
        train_file=train_file,
        val_file=val_file if val_file.exists() else None,
        output_dir=Path(args.out),
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        load_in_4bit=not args.no_4bit,
        seed=args.seed,
        max_samples=args.limit_samples or None,
    )
    print(f"[train] base={config.base_model}")
    print(f"[train] train_file={train_file} → out={config.output_dir}")
    report = train_lora(config)
    print("[train] 完成：")
    for key in ("samples", "train_loss", "elapsed_s", "use_4bit", "device", "trainable_params"):
        print(f"  {key}: {report[key]}")
    print(f"  adapter: {report['adapter_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
