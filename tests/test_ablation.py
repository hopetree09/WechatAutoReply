"""消融实验工具函数单测（不调用模型）。"""

from __future__ import annotations

import json

import pytest

from wagent.eval.ablation import (
    clean_line,
    has_memory,
    load_cases,
    memory_block_for,
    seed_memory_store,
)
from wagent.memory import MemoryStore


def write_eval_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )


def make_row(latest: str, reply: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": f"最近聊天记录：\n对方: {latest}\n\n对方刚发来：{latest}\n\n请回复。",
            },
            {"role": "assistant", "content": reply},
        ]
    }


def test_load_cases_extracts_latest_and_reference(tmp_path) -> None:
    path = tmp_path / "val.jsonl"
    write_eval_jsonl(path, [make_row("在吗", "在的"), make_row("搬家了吗", "搬完了")])
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].latest == "在吗"
    assert cases[0].reference == "在的"
    assert cases[1].user_content.startswith("最近聊天记录")


def test_load_cases_limit_and_skip_invalid(tmp_path) -> None:
    path = tmp_path / "val.jsonl"
    bad = {"messages": [{"role": "user", "content": "x"}]}
    write_eval_jsonl(path, [make_row("a", "b"), bad, make_row("c", "d")])
    assert len(load_cases(path, limit=1)) == 1
    assert len(load_cases(path, limit=10)) == 2


def test_seed_memory_store_creates_all_layers() -> None:
    store = MemoryStore(":memory:")
    stats = seed_memory_store(store, style_samples=["行吧 先这样"])
    assert stats["profiles"] == 2
    assert stats["facts"] == 3
    assert stats["episodes"] == 2
    assert stats["style_samples"] == 1
    assert store.get_profile("张伟") is not None
    assert store.get_profile("__self__") is not None


def test_memory_block_for_contains_sections() -> None:
    store = MemoryStore(":memory:")
    seed_memory_store(store, style_samples=["行吧 先这样"])
    block = memory_block_for(store, "张伟", "搬家的事定了吗")
    assert "【人物画像】" in block
    assert "【我记得的事】" in block
    assert "搬家" in block
    assert has_memory(block)


def test_memory_block_empty_for_unknown_person() -> None:
    store = MemoryStore(":memory:")
    block = memory_block_for(store, "陌生人", "在吗")
    assert block == ""
    assert not has_memory(block)


def test_has_memory_detects_sections() -> None:
    assert has_memory("【最近发生】- 11-01 打球")
    assert not has_memory("只有人设和风格规则")


def test_clean_line_collapses_whitespace() -> None:
    assert clean_line("  行吧\n\n先这样 ") == "行吧 先这样"


def test_load_cases_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope.jsonl")
