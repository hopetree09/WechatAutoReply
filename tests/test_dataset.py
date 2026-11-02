"""LoRA 训练集构造单测。"""

from __future__ import annotations

from wagent.memory.store import MemoryStore
from wagent.memory.types import MessageRecord
from wagent.train.dataset import (
    TrainingSample,
    build_samples,
    export_dataset,
    split_samples,
)


def rec(speaker: str, text: str, i: int, person: str = "张三") -> MessageRecord:
    return MessageRecord(
        id=f"{person}_msg_{i}",
        person=person,
        speaker=speaker,
        text=text,
        ts=f"2026-11-01T10:{i:02d}:00+08:00",
    )


def records() -> list[MessageRecord]:
    return [
        rec("other", "在吗", 0),
        rec("me", "在的", 1),
        rec("other", "今晚有空吗", 2),
        rec("me", "有啊 八点老地方", 3),
    ]


def test_build_samples_only_for_my_replies_after_other() -> None:
    samples = build_samples(records())
    assert [s.reply for s in samples] == ["在的", "有啊 八点老地方"]
    assert [s.latest for s in samples] == ["在吗", "今晚有空吗"]
    # 第二个样本上下文包含前三条
    assert len(samples[1].context) == 3


def test_build_samples_skips_reply_without_other_context() -> None:
    # 开头连续两条我的消息（没有对方上下文）→ 只有第一条能成样本
    data = [rec("me", "早", 0), rec("me", "我起了", 1)]
    assert build_samples(data) == []


def test_build_samples_window_and_length_filter() -> None:
    many = [rec("other", "嗯", 0)]
    many += [rec("me", f"回复{i}", i + 1) for i in range(10)]
    samples = build_samples(many, window=3, max_reply_chars=4)
    assert all(len(s.context) <= 3 for s in samples)
    assert all(len(s.reply) <= 4 for s in samples)


def test_build_samples_dedupes_identical_tail_and_reply() -> None:
    data = [
        rec("other", "在吗", 0),
        rec("me", "在的", 1),
        rec("other", "在吗", 2),
        rec("me", "在的", 3),
    ]
    assert len(build_samples(data)) == 1


def test_chatml_structure() -> None:
    sample = build_samples(records())[1]
    row = sample.to_chatml()
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][-1]["content"] == "有啊 八点老地方"
    assert "对方: 今晚有空吗" in row["messages"][1]["content"]
    assert "请以我的口吻回复" in row["messages"][1]["content"]


def test_split_sizes_and_determinism() -> None:
    samples = [TrainingSample(context=records()[:2], reply=f"回复{i}") for i in range(20)]
    train_a, val_a = split_samples(samples, val_ratio=0.2, seed=1)
    train_b, val_b = split_samples(samples, val_ratio=0.2, seed=1)
    assert [s.reply for s in train_a] == [s.reply for s in train_b]
    assert len(val_a) == 4 and len(train_a) == 16
    assert val_a == val_b
    assert train_a + val_a != [] and len(set(s.reply for s in train_a + val_a)) == 20


def test_split_small_dataset_keeps_val_empty() -> None:
    samples = [TrainingSample(context=records()[:2], reply=f"回复{i}") for i in range(5)]
    train, val = split_samples(samples)
    assert len(train) == 5 and val == []


def test_export_dataset_writes_jsonl(tmp_path) -> None:
    store = MemoryStore(":memory:")
    store.add_messages(records())
    stats = export_dataset(store, tmp_path / "out")
    assert stats["samples"] == 2
    assert stats["train"] == 2 and stats["val"] == 0
    train_lines = (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 2
    assert '"assistant"' in train_lines[0]


def test_export_dataset_filters_person(tmp_path) -> None:
    store = MemoryStore(":memory:")
    store.add_messages(records())
    store.add_messages(
        [
            rec("other", "另一个人的消息", 0, person="李四"),
            rec("me", "另一个人的回复", 1, person="李四"),
        ]
    )
    stats = export_dataset(store, tmp_path / "out", person="李四")
    assert stats["people"] == ["李四"]
    assert stats["samples"] == 1
