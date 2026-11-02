"""训练侧纯逻辑单测：合成数据、ChatML 读写、配置序列化（不需要 torch）。"""

from __future__ import annotations

import json

import pytest

from wagent.train.dataset import build_samples, split_samples, write_jsonl
from wagent.train.lora import (
    LoraConfig,
    format_chatml_fallback,
    read_chatml,
    texts_from_rows,
)
from wagent.train.synth import SynthConfig, generate_records

# ---------------------------------------------------------------- 合成数据


def test_synth_is_deterministic() -> None:
    a = generate_records(SynthConfig(exchanges=20, seed=3))
    b = generate_records(SynthConfig(exchanges=20, seed=3))
    c = generate_records(SynthConfig(exchanges=20, seed=4))
    key = lambda r: (r.speaker, r.text, r.ts)  # noqa: E731
    assert [key(r) for r in a] == [key(r) for r in b]
    assert [key(r) for r in a] != [key(r) for r in c]


def test_synth_shape_and_time_order() -> None:
    records = generate_records(SynthConfig(exchanges=30, seed=1))
    assert len(records) > 30
    assert {r.person for r in records} == {"张伟"}
    assert {r.speaker for r in records} == {"me", "other"}
    # 至少 30 组「对方 → 我」的配对，可用于构造训练样本
    pairs = sum(1 for i in range(1, len(records))
                if records[i - 1].speaker == "other" and records[i].speaker == "me")
    assert pairs >= 30


def test_synth_builds_training_samples() -> None:
    records = generate_records(SynthConfig(exchanges=30, seed=2))
    samples = build_samples(records, window=6)
    assert len(samples) >= 20
    train, val = split_samples(samples, val_ratio=0.1, seed=2)
    assert len(train) + len(val) == len(samples)
    assert len(val) >= 1  # samples >= 10 时应切出验证集
    row = samples[0].to_chatml()
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]


# ---------------------------------------------------------------- ChatML 读写


def test_chatml_roundtrip(tmp_path) -> None:
    records = generate_records(SynthConfig(exchanges=10, seed=5))
    samples = build_samples(records)
    rows = [s.to_chatml() for s in samples]
    path = tmp_path / "train.jsonl"
    assert write_jsonl(rows, path) == len(rows)
    loaded = read_chatml(path)
    assert len(loaded) == len(rows)
    assert loaded[0]["messages"][-1]["role"] == "assistant"


def test_read_chatml_rejects_bad_rows(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="assistant"):
        read_chatml(path)


def test_read_chatml_rejects_empty(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="数据集为空"):
        read_chatml(path)


def test_read_chatml_rejects_missing_messages(tmp_path) -> None:
    path = tmp_path / "bad2.jsonl"
    path.write_text('{"foo": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少 messages"):
        read_chatml(path)


def test_format_fallback_and_texts() -> None:
    rows = [{"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]}]
    text = format_chatml_fallback(rows[0]["messages"])
    assert "<|system|>" in text and "<|assistant|>" in text
    assert texts_from_rows(rows) == [text]


def test_texts_from_rows_uses_tokenizer_template() -> None:
    class FakeTokenizer:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False):
            assert tokenize is False
            return f"TPL:{messages[-1]['content']}"

    rows = [{"messages": [{"role": "assistant", "content": "好的"}]}]
    assert texts_from_rows(rows, FakeTokenizer()) == ["TPL:好的"]


# ---------------------------------------------------------------- 配置


def test_lora_config_serializable(tmp_path) -> None:
    config = LoraConfig(
        base_model="Qwen2.5-3B-Instruct",
        train_file=tmp_path / "train.jsonl",
        output_dir=tmp_path / "adapter",
        val_file=tmp_path / "val.jsonl",
    )
    data = config.to_dict()
    assert data["train_file"].endswith("train.jsonl")
    assert isinstance(data["epochs"], float)
    json.dumps(data, ensure_ascii=False)  # 可序列化
