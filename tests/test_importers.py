"""CSV 导入 + 时间戳解析 + 滚动采史 单测。"""

from __future__ import annotations

from datetime import datetime

import pytest

from wagent.importers.csv_importer import ColumnMapping, CsvImporter, parse_timestamp
from wagent.importers.scroll_harvester import (
    ScrollHarvester,
    merge_frames,
)
from wagent.perception.tsutil import resolve_ts_text
from wagent.perception.types import Box, Message, Speaker

NOW = datetime.fromisoformat("2026-11-02T12:00:00+08:00")


def msg(speaker: Speaker, text: str, ts_text: str = "") -> Message:
    return Message(speaker=speaker, text=text, box=Box(0, 0, 10, 10), ts_text=ts_text)


# ---------------------------------------------------------------- CSV


def test_csv_import_chinese_headers(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    path.write_text(
        "时间,内容,是否发送,类型,好友\n"
        "2026-11-01 10:00:00,在吗,0,1,张三\n"
        "2026-11-01 10:01:00,在的 咋了,1,1,张三\n"
        "2026-11-01 10:02:00,[图片],0,3,张三\n",
        encoding="utf-8-sig",
    )
    result = CsvImporter(path=path).run()
    assert result.total_rows == 3
    assert result.imported == 2
    assert result.skipped_media == 1
    assert [m.speaker for m in result.messages] == ["other", "me"]
    assert result.messages[0].person == "张三"
    assert result.messages[0].ts.startswith("2026-11-01T10:00:00")


def test_csv_import_english_headers_and_person_override(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    path.write_text(
        "StrTime,StrContent,IsSender,Type,StrTalker\n"
        "2026/11/01 10:00:00,hello,0,1,whatever\n"
        "2026/11/01 10:01:00,hi,1,1,whatever\n",
        encoding="utf-8",
    )
    result = CsvImporter(path=path, person="李四").run()
    assert result.imported == 2
    assert all(m.person == "李四" for m in result.messages)


def test_csv_import_gbk_encoding(tmp_path) -> None:
    path = tmp_path / "gbk.csv"
    path.write_text("时间,内容,是否发送\n2026-11-01 10:00:00,在吗,0\n", encoding="gbk")
    result = CsvImporter(path=path, person="张三").run()
    assert result.imported == 1
    assert result.messages[0].text == "在吗"


def test_csv_without_type_column_filters_media_and_timestamps(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    path.write_text(
        "时间,内容,是否发送\n"
        "2026-11-01 10:00:00,在吗,0\n"
        "2026-11-01 10:01:00,[图片],0\n"
        "2026-11-01 10:02:00,昨天 21:03,0\n",
        encoding="utf-8",
    )
    result = CsvImporter(path=path, person="张三").run()
    assert result.imported == 1
    assert result.messages[0].text == "在吗"


def test_csv_dedupes_identical_rows(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    path.write_text(
        "时间,内容,是否发送\n"
        "2026-11-01 10:00:00,在吗,0\n"
        "2026-11-01 10:00:00,在吗,0\n",
        encoding="utf-8",
    )
    result = CsvImporter(path=path, person="张三").run()
    assert result.imported == 1


def test_csv_missing_columns_raises(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必要列"):
        CsvImporter(path=path).run()


def test_csv_bad_timestamp_counted(tmp_path) -> None:
    path = tmp_path / "chat.csv"
    path.write_text("时间,内容,是否发送\nnot-a-time,在吗,0\n", encoding="utf-8")
    result = CsvImporter(path=path, person="张三").run()
    assert result.imported == 0
    assert result.skipped_time == 1


def test_parse_timestamp_formats() -> None:
    assert parse_timestamp("2026-11-01 10:00:00").startswith("2026-11-01T10:00:00")
    assert parse_timestamp("2026/11/01 10:00").startswith("2026-11-01T10:00:00")
    assert parse_timestamp("2026年11月1日 10:00").startswith("2026-11-01T10:00:00")
    assert parse_timestamp("") is None
    assert parse_timestamp("随便一句话") is None


def test_column_mapping_autodetect() -> None:
    mapping = ColumnMapping.autodetect(["StrTime", "StrContent", "IsSender", "Type"])
    assert mapping.ts == "StrTime"
    assert mapping.text == "StrContent"
    assert mapping.is_sender == "IsSender"


# ---------------------------------------------------------------- 时间戳文本


def test_resolve_ts_text() -> None:
    assert resolve_ts_text("21:03", NOW) == "2026-11-02T21:03:00+08:00"
    assert resolve_ts_text("昨天 21:03", NOW) == "2026-11-01T21:03:00+08:00"
    assert resolve_ts_text("前天 8:05", NOW) == "2026-10-31T08:05:00+08:00"
    assert resolve_ts_text("2026年10月20日 08:21", NOW) == "2026-10-20T08:21:00+08:00"
    assert resolve_ts_text("10月20日 08:21", NOW) == "2026-10-20T08:21:00+08:00"
    assert resolve_ts_text("下午3:05", NOW) == "2026-11-02T15:05:00+08:00"


def test_resolve_ts_text_weekday_and_cross_year() -> None:
    # 2026-11-02 是周一，最近的周五是 10-30
    assert resolve_ts_text("星期五 15:15", NOW) == "2026-10-30T15:15:00+08:00"
    # 12 月 20 日在 11-02 看来是未来 → 归到去年
    assert resolve_ts_text("12月20日", NOW) == "2025-12-20T00:00:00+08:00"


def test_resolve_ts_text_invalid() -> None:
    assert resolve_ts_text("", NOW) is None
    assert resolve_ts_text("对方撤回了一条消息", NOW) is None


# ---------------------------------------------------------------- 滚动拼接


def test_merge_frames_overlap() -> None:
    """向上滚动采史：新帧 = 更早消息 + 与已有序列重叠的顶部。"""
    existing = [msg(Speaker.ME, "B"), msg(Speaker.OTHER, "C")]
    frame = [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B"), msg(Speaker.OTHER, "C")]
    merged, new, overlap = merge_frames(existing, frame)
    assert [m.text for m in merged] == ["A", "B", "C"]
    assert new == 1 and overlap == 2


def test_merge_frames_full_duplicate() -> None:
    existing = [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B")]
    merged, new, overlap = merge_frames(existing, [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B")])
    assert merged == existing and new == 0 and overlap == 2


def test_merge_frames_no_overlap_prepends_older() -> None:
    existing = [msg(Speaker.OTHER, "C"), msg(Speaker.ME, "D")]
    frame = [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B")]
    merged, new, overlap = merge_frames(existing, frame)
    assert [m.text for m in merged] == ["A", "B", "C", "D"]
    assert new == 2 and overlap == 0


def test_merge_frames_empty_cases() -> None:
    assert merge_frames([], []) == ([], 0, 0)
    frame = [msg(Speaker.OTHER, "A")]
    assert merge_frames([], frame) == (frame, 1, 0)
    assert merge_frames(frame, []) == (frame, 0, 0)


def test_scroll_harvester_stitches_and_stops_when_stagnant() -> None:
    frames = [
        [msg(Speaker.OTHER, "C"), msg(Speaker.ME, "D")],
        [msg(Speaker.ME, "B"), msg(Speaker.OTHER, "C"), msg(Speaker.ME, "D")],
        [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B"), msg(Speaker.OTHER, "C"), msg(Speaker.ME, "D")],
        [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B"), msg(Speaker.OTHER, "C"), msg(Speaker.ME, "D")],
        [msg(Speaker.OTHER, "A"), msg(Speaker.ME, "B"), msg(Speaker.OTHER, "C"), msg(Speaker.ME, "D")],
    ]
    index = {"i": 0}
    scrolls = {"n": 0}

    def grab():
        frame = frames[min(index["i"], len(frames) - 1)]
        index["i"] += 1
        return frame

    def scroll():
        scrolls["n"] += 1

    harvester = ScrollHarvester(
        grab_messages=grab,
        scroll_up=scroll,
        person="张三",
        max_frames=10,
        max_stagnant=2,
    )
    result = harvester.harvest()
    assert [m.text for m in result.messages] == ["A", "B", "C", "D"]
    assert result.stopped == "stagnant"
    assert result.frames >= 3
    assert scrolls["n"] >= 2


def test_scroll_harvester_converts_timestamps() -> None:
    frames = [[msg(Speaker.OTHER, "在吗", ts_text="昨天 21:03"), msg(Speaker.ME, "在的")]]

    harvester = ScrollHarvester(
        grab_messages=lambda: frames[0],
        scroll_up=lambda: None,
        person="张三",
        max_frames=1,
        now=lambda: NOW,
    )
    result = harvester.harvest()
    assert result.messages[0].ts == "2026-11-01T21:03:00+08:00"
    assert result.messages[1].ts  # 没有时间戳时用基准时间兜底


def test_scroll_harvester_empty_frame_stops() -> None:
    harvester = ScrollHarvester(
        grab_messages=lambda: [],
        scroll_up=lambda: None,
        person="张三",
        max_frames=5,
    )
    result = harvester.harvest()
    assert result.stopped == "empty"
    assert result.messages == []
