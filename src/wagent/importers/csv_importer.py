"""CSV 聊天记录导入器。

兼容常见的微信导出工具字段命名（中英文都做了别名），自动识别列名与编码；
无法解析的行会计入统计而不是中断导入。

典型字段：
    StrTime / 时间 / createTime      → 时间
    StrContent / 内容 / content      → 文本
    IsSender / 是否发送               → 1=我，0=对方
    Type / MsgType                   → 1=文本，其余多为图片/语音/表情
    StrTalker / 好友 / NickName       → 联系人（1v1 时可用 --person 覆盖）
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from wagent.importers.base import ImportResult
from wagent.memory.types import MessageRecord, now_iso
from wagent.perception.parser import classify_text

TS_ALIASES = ("StrTime", "时间", "time", "createTime", "CreateTime", "TalkTime", "Date", "datetime")
TEXT_ALIASES = ("StrContent", "内容", "content", "msg", "Message", "StrMsg", "text", "消息")
SENDER_ALIASES = ("IsSender", "是否发送", "isSender", "is_sender", "SenderIsMe")
SPEAKER_ALIASES = ("Sender", "NickName", "发送人", "speaker", "Nick", "StrSender")
TYPE_ALIASES = ("Type", "MsgType", "消息类型", "类型", "msgType", "Type_")
PERSON_ALIASES = ("StrTalker", "Talker", "好友", "contact", "person", "ChatName")

# 微信 3.x/4.x 里 Type=1 是文本，其余为媒体/系统
TEXT_TYPES = {"1", "text", "txt", "文本", "文字"}

_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y年%m月%d日 %H:%M",
    "%Y-%m-%d",
)


def parse_timestamp(raw: str) -> str | None:
    """把各种导出格式的时间统一成带时区的 ISO 字符串。"""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for fmt in _TS_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.isoformat(timespec="seconds")


def _find_column(headers: Sequence[str], aliases: Sequence[str]) -> str | None:
    lowered = {h.strip().lower(): h for h in headers}
    for alias in aliases:
        if alias.strip().lower() in lowered:
            return lowered[alias.strip().lower()]
    return None


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """CSV 列映射；不传时按别名自动识别。"""

    ts: str | None = None
    text: str | None = None
    is_sender: str | None = None
    speaker: str | None = None
    msg_type: str | None = None
    person: str | None = None

    @classmethod
    def autodetect(cls, headers: Sequence[str]) -> ColumnMapping:
        return cls(
            ts=_find_column(headers, TS_ALIASES),
            text=_find_column(headers, TEXT_ALIASES),
            is_sender=_find_column(headers, SENDER_ALIASES),
            speaker=_find_column(headers, SPEAKER_ALIASES),
            msg_type=_find_column(headers, TYPE_ALIASES),
            person=_find_column(headers, PERSON_ALIASES),
        )

    def validate(self, headers: Sequence[str]) -> None:
        missing = []
        if not self.ts or self.ts not in headers:
            missing.append("时间(StrTime/时间)")
        if not self.text or self.text not in headers:
            missing.append("内容(StrContent/内容)")
        if missing:
            raise ValueError(
                "CSV 缺少必要列：" + "、".join(missing) + f"；实际列名：{list(headers)}"
            )


def read_text_auto(path: Path) -> str:
    """按常见编码依次尝试读取（导出工具常给 utf-8-sig 或 gbk）。"""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


@dataclass
class CsvImporter:
    """把 CSV 转成 :class:`MessageRecord` 列表。"""

    path: Path
    person: str = ""
    mapping: ColumnMapping | None = None
    me_names: tuple[str, ...] = ("我", "me", "self", "myself")
    encoding: str | None = None
    default_ts: str = field(default_factory=now_iso)

    def _rows(self) -> tuple[list[dict[str, str]], list[str]]:
        text = (
            self.path.read_text(encoding=self.encoding)
            if self.encoding
            else read_text_auto(self.path)
        )
        reader = csv.DictReader(io.StringIO(text))
        headers = [h for h in (reader.fieldnames or []) if h is not None]
        rows = [row for row in reader]
        return rows, headers

    def run(self) -> ImportResult:
        rows, headers = self._rows()
        mapping = self.mapping or ColumnMapping.autodetect(headers)
        mapping.validate(headers)
        result = ImportResult(total_rows=len(rows), messages=[])

        person = self.person.strip()
        for row in rows:
            if not person and mapping.person:
                person = (row.get(mapping.person) or "").strip()
            record = self._to_record(row, mapping, person, result)
            if record is not None:
                result.messages.append(record)

        # 同一时间戳 + 内容的重复行去重（部分导出会重复）
        seen: set[tuple[str, str]] = set()
        deduped: list[MessageRecord] = []
        for rec in result.messages:
            key = (rec.ts, rec.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        result.messages = sorted(deduped, key=lambda r: r.ts)
        return result

    def _to_record(
        self,
        row: dict[str, str],
        mapping: ColumnMapping,
        person: str,
        result: ImportResult,
    ) -> MessageRecord | None:
        text = (row.get(mapping.text or "") or "").strip()
        if not text:
            result.skipped_empty += 1
            return None

        if mapping.msg_type:
            msg_type = (row.get(mapping.msg_type) or "").strip().lower()
            if msg_type and msg_type not in TEXT_TYPES:
                result.skipped_media += 1
                return None

        # 没有类型列时按文本特征过滤（[图片]/时间戳/系统提示）
        if not mapping.msg_type and classify_text(text) != "message":
            result.skipped_empty += 1
            return None

        ts = parse_timestamp(row.get(mapping.ts or "") or "")
        if ts is None:
            result.skipped_time += 1
            return None

        speaker = self._speaker_of(row, mapping)
        return MessageRecord(
            person=person or "unknown",
            speaker=speaker,
            text=text,
            ts=ts,
            session=person,
            source="import",
        )

    def _speaker_of(self, row: dict[str, str], mapping: ColumnMapping) -> str:
        if mapping.is_sender:
            raw = (row.get(mapping.is_sender) or "").strip().lower()
            if raw in {"1", "true", "yes", "是"}:
                return "me"
            if raw in {"0", "false", "no", "否"}:
                return "other"
        if mapping.speaker:
            who = (row.get(mapping.speaker) or "").strip()
            if who.lower() in {name.lower() for name in self.me_names}:
                return "me"
            if who:
                return "other"
        return "other"


def iter_csv_records(path: Path, *, person: str = "") -> Iterator[MessageRecord]:
    """便捷函数：直接迭代导入结果。"""
    yield from CsvImporter(path=path, person=person).run().messages
