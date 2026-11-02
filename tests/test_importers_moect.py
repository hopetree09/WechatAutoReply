"""Moect JSON(L) / CSV 扩展列 / 导出文件探测 单测。"""

from __future__ import annotations

import json

import pytest

from wagent.importers.csv_importer import CsvImporter
from wagent.importers.detect import peek_export
from wagent.importers.moect_json import MoectJsonImporter


def moect_msg(
    *,
    ts: str = "2026-10-28 20:00:00",
    sender: str = "张伟",
    is_sender: bool = False,
    type_name: str = "文本",
    text: str = "在吗",
    server_id: str = "123",
    kind: str = "text",
) -> dict:
    return {
        "time_str": ts,
        "sender": sender,
        "sender_wxid": "wxid_me" if is_sender else "wxid_other",
        "is_sender": is_sender,
        "type_name": type_name,
        "text": text,
        "server_id": server_id,
        "kind": kind,
    }


# ---------------------------------------------------------------- JSONL


def write_jsonl(path, messages: list[dict], metadata: dict | None = None) -> None:
    lines = [json.dumps({"metadata": metadata or {"session": "wxid_other", "name": "张伟"}}, ensure_ascii=False)]
    lines += [json.dumps(msg, ensure_ascii=False) for msg in messages]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_jsonl_import_basic(tmp_path) -> None:
    path = tmp_path / "张伟.jsonl"
    write_jsonl(
        path,
        [
            moect_msg(ts="2026-10-28 20:00:00", is_sender=False, text="在吗"),
            moect_msg(ts="2026-10-28 20:01:00", is_sender=True, text="在的 咋了"),
        ],
    )
    result = MoectJsonImporter(path).run()
    assert result.total_rows == 2
    assert result.imported == 2
    assert [m.speaker for m in result.messages] == ["other", "me"]
    assert result.messages[0].person == "张伟"
    assert result.messages[0].ts.startswith("2026-10-28T20:00:00")
    assert result.messages[0].id  # server_id → 稳定 id


def test_jsonl_media_skip_and_keep(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(
        path,
        [
            moect_msg(text="在吗"),
            moect_msg(type_name="图片", text="", kind="image", server_id="2"),
            moect_msg(type_name="系统消息", text="你已添加了张伟", kind="system", server_id="3"),
        ],
    )
    plain = MoectJsonImporter(path).run()
    assert plain.imported == 1
    assert plain.skipped_media == 1
    assert plain.skipped_empty == 1

    keep = MoectJsonImporter(path, keep_media=True).run()
    assert keep.imported == 2
    assert any("[图片]" in m.text for m in keep.messages)


def test_jsonl_person_override_and_dedup(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(
        path,
        [
            moect_msg(ts="2026-10-28 20:00:00", text="在吗"),
            moect_msg(ts="2026-10-28 20:00:00", text="在吗"),  # 完全重复
            moect_msg(ts="2026-10-28 20:01:00", text="喂"),
        ],
    )
    result = MoectJsonImporter(path, person="张三").run()
    assert result.imported == 2
    assert all(m.person == "张三" for m in result.messages)


def test_json_import_dict_payload(tmp_path) -> None:
    path = tmp_path / "x.json"
    payload = {
        "metadata": {"session": "wxid_x", "name": "李四"},
        "messages": [
            moect_msg(ts="2026-10-28 20:00:00", text="你好"),
            moect_msg(ts="2026-10-28 20:01:00", is_sender=True, text="嗯"),
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = MoectJsonImporter(path).run()
    assert result.imported == 2
    assert result.messages[0].person == "李四"


def test_jsonl_skips_bad_lines(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(
        '{"metadata": {"name": "张伟"}}\n'
        "not json\n"
        + json.dumps(moect_msg(text="在吗"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result = MoectJsonImporter(path).run()
    assert result.imported == 1


# ---------------------------------------------------------------- CSV 扩展


def test_csv_with_moect_self_column(tmp_path) -> None:
    """Moect 的 wx4 CSV 带「是否我发送」列。"""
    path = tmp_path / "x.csv"
    path.write_text(
        "时间,会话,发送者,是否我发送,类型,内容,server_id\n"
        "2026-10-28 20:00:00,张伟,张伟,否,文本,在吗,1\n"
        "2026-10-28 20:01:00,张伟,我的昵称,是,文本,在的,2\n"
        "2026-10-28 20:02:00,张伟,张伟,否,图片, ,3\n",
        encoding="utf-8-sig",
    )
    result = CsvImporter(path=path).run()
    assert result.imported == 2
    assert [m.speaker for m in result.messages] == ["other", "me"]
    assert result.skipped_media == 1
    assert not result.warnings


def test_csv_without_self_column_warns(tmp_path) -> None:
    """Moect 的 core CSV 没有「是否我发送」列 → 必须给 --me，否则告警。"""
    path = tmp_path / "x.csv"
    path.write_text(
        "时间,会话,发送者,类型,内容,消息ID\n"
        "2026-10-28 20:00:00,张伟,张伟,文本,在吗,1\n"
        "2026-10-28 20:01:00,张伟,我的昵称,文本,在的,2\n",
        encoding="utf-8-sig",
    )
    plain = CsvImporter(path=path).run()
    assert plain.imported == 2
    assert plain.warnings  # 提醒所有消息都被当成对方
    assert all(m.speaker == "other" for m in plain.messages)

    fixed = CsvImporter(path=path, me_names=("我的昵称",)).run()
    assert [m.speaker for m in fixed.messages] == ["other", "me"]
    assert not fixed.warnings


# ---------------------------------------------------------------- 探测


def test_peek_csv(tmp_path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("时间,内容,是否我发送\n2026-10-28 20:00:00,在吗,否\n", encoding="utf-8")
    peek = peek_export(path)
    assert peek.kind == "csv"
    assert peek.message_count == 1
    assert "import_chat csv" in peek.suggested_command(person="张三")


def test_peek_jsonl(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [moect_msg()])
    peek = peek_export(path)
    assert peek.kind == "jsonl"
    assert peek.message_count == 1
    assert "--keep-media" not in peek.suggested_command(person="张伟")
    assert "import_chat json" in peek.suggested_command(person="张伟")


def test_peek_json(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps({"metadata": {}, "messages": [moect_msg(), moect_msg(server_id="2")]}, ensure_ascii=False),
        encoding="utf-8",
    )
    peek = peek_export(path)
    assert peek.kind == "json"
    assert peek.message_count == 2


def test_peek_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        peek_export(tmp_path / "nope.csv")


def test_peek_csv_missing_columns_notes(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    peek = peek_export(path)
    assert peek.kind == "csv"
    assert any("缺少时间列" in note for note in peek.notes)
    assert any("缺少内容列" in note for note in peek.notes)
