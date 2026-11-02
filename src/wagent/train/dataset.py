"""从记忆库构造 LoRA 训练集（Q/A 对 → ChatML JSONL）。

训练目标是「学口吻」，不是「学知识」：
- 输入：最近若干轮聊天记录 + 对方最新一条（带角色标注）
- 输出：我当时真实回复的那句话

所以样本只保留「我」真实说过的话，且上下文里必须有对方的消息；
同一个（上下文尾部, 回复）只保留一份，避免模型记住某几条高频回复。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from wagent.memory.store import MemoryStore
from wagent.memory.types import MessageRecord

DEFAULT_SYSTEM = "你是「我」本人，正在微信里和熟人聊天，用我平时的口吻简短回复。"


@dataclass(frozen=True, slots=True)
class TrainingSample:
    context: list[MessageRecord]
    reply: str

    @property
    def latest(self) -> str:
        for message in reversed(self.context):
            if message.speaker == "other":
                return message.text
        return ""

    def to_chatml(self, *, system: str = DEFAULT_SYSTEM) -> dict:
        history = "\n".join(
            f"{'我' if m.speaker == 'me' else '对方'}: {m.text}" for m in self.context
        )
        user = (
            f"最近聊天记录：\n{history}\n\n"
            f"对方刚发来：{self.latest}\n\n"
            "请以我的口吻回复（只输出回复内容）："
        )
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": self.reply},
            ]
        }


def build_samples(
    records: list[MessageRecord],
    *,
    window: int = 6,
    min_reply_chars: int = 1,
    max_reply_chars: int = 120,
) -> list[TrainingSample]:
    """按时间顺序扫描，把「我」的每条回复变成一条训练样本。"""
    samples: list[TrainingSample] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if record.speaker != "me":
            continue
        reply = record.text.strip()
        if not (min_reply_chars <= len(reply) <= max_reply_chars):
            continue
        context = records[max(0, index - window) : index]
        if not any(message.speaker == "other" for message in context):
            continue
        latest = next(
            (m.text for m in reversed(context) if m.speaker == "other"), ""
        )
        key = (latest, reply)
        if key in seen:
            continue
        seen.add(key)
        samples.append(TrainingSample(context=context, reply=reply))
    return samples


def split_samples(
    samples: list[TrainingSample], *, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """按会话/时间无关的随机切分（小数据集，先保证可复现）。"""
    if not samples:
        return [], []
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) >= 10 else 0
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(rows: list[dict], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def export_dataset(
    store: MemoryStore,
    out_dir: Path,
    *,
    person: str | None = None,
    window: int = 6,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """从记忆库导出一份可直接喂给 LLaMA-Factory/Unsloth 的数据集。"""
    people = [person] if person else _list_people(store)
    all_samples: list[TrainingSample] = []
    for who in people:
        records = store.recent_messages(who, limit=100_000)
        all_samples.extend(build_samples(records, window=window))
    train, val = split_samples(all_samples, val_ratio=val_ratio, seed=seed)
    train_rows = [sample.to_chatml() for sample in train]
    val_rows = [sample.to_chatml() for sample in val]
    out_dir = Path(out_dir)
    write_jsonl(train_rows, out_dir / "train.jsonl")
    write_jsonl(val_rows, out_dir / "val.jsonl")
    lengths = [len(sample.reply) for sample in all_samples]
    return {
        "people": people,
        "samples": len(all_samples),
        "train": len(train_rows),
        "val": len(val_rows),
        "avg_reply_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "max_reply_chars": max(lengths) if lengths else 0,
    }


def _list_people(store: MemoryStore) -> list[str]:
    rows = store.conn.execute("SELECT DISTINCT person FROM messages ORDER BY person").fetchall()
    return [row["person"] for row in rows]
