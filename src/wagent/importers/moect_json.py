"""Moect（微信 4.x 导出工具）JSON / JSONL 导入器。

为什么优先推荐 JSONL 而不是 CSV：
- Moect 的 CSV（v1.2.0 的 core 导出路径）表头是
  ``时间,会话,发送者,类型,内容,消息ID``，**没有"是否我发送"列**，
  只能靠"发送者 == 我自己的昵称"去猜；
- 而 JSON/JSONL 里的消息对象带 ``is_sender`` / ``sender_wxid`` / ``sender`` /
  ``type_name`` / ``server_id`` 等完整字段，说话人判定是确定的。

支持两种文件形态：
- ``.jsonl``：第一行是 ``{"metadata": {...}}``，之后每行一条消息；
- ``.json``：``{"metadata": {...}, "messages": [...]}`` 或直接消息数组。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from wagent.importers.base import ImportResult
from wagent.importers.csv_importer import parse_timestamp
from wagent.memory.types import MessageRecord

# 这些类型不是聊天内容，直接跳过（系统提示/好友请求/通话等）
SYSTEM_TYPES = frozenset({"系统消息", "好友请求", "音视频通话", "状态", "连线"})
# 有媒体占位意义（keep_media=True 时保留为 [类型] 占位）
MEDIA_TYPES = frozenset(
    {"图片", "语音", "视频", "动画表情", "文件", "位置", "应用消息", "小视频", "名片", "音视频通话", "引用消息"}
)
TEXT_TYPE = "文本"


@dataclass(frozen=True, slots=True)
class MoectMessage:
    """Moect 消息对象的最小字段集。"""

    data: dict


def _load_payload(path: Path) -> tuple[dict, list[dict]]:
    """返回 ``(metadata, messages)``；兼容 .json / .jsonl。"""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        metadata: dict = {}
        messages: list[dict] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "messages" in item:
                metadata.update(item.get("metadata") or {})
                messages.extend(item["messages"])
            elif isinstance(item, dict) and "metadata" in item:
                metadata.update(item["metadata"] or {})
            elif isinstance(item, dict):
                messages.append(item)
        return metadata, messages

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return {}, [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return payload.get("metadata") or {}, [
            item for item in payload.get("messages") or [] if isinstance(item, dict)
        ]
    return {}, []


def _epoch_to_iso(value) -> str | None:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def _message_text(data: dict, *, keep_media: bool) -> tuple[str, str]:
    """返回 ``(文本, 处理结果)``；处理结果为 keep / media_skip / empty。"""
    type_name = str(data.get("type_name") or "")
    kind = str(data.get("kind") or "")
    text = str(data.get("text") or "").strip()
    if type_name in SYSTEM_TYPES or kind == "system":
        return "", "system"
    is_text = type_name == TEXT_TYPE or kind == "text"
    if is_text:
        return (text, "keep") if text else ("", "empty")
    if keep_media:
        label = text or (f"[{type_name}]" if type_name else "[非文本]")
        return label, "keep"
    return "", "media_skip"


@dataclass
class MoectJsonImporter:
    """把 Moect 的 JSON/JSONL 导出转成统一 ``MessageRecord``。"""

    path: Path
    person: str = ""
    session: str = ""
    keep_media: bool = False
    limit: int | None = None

    def run(self) -> ImportResult:
        metadata, raw_messages = _load_payload(Path(self.path))
        result = ImportResult(total_rows=len(raw_messages))
        person = (
            self.person.strip()
            or str(metadata.get("name") or "").strip()
            or str(raw_messages[0].get("session") or "").strip() if raw_messages else self.person
        )
        person = person or "unknown"
        session = self.session or str(metadata.get("session") or "") or person

        records: list[MessageRecord] = []
        seen: set[tuple[str, str, str]] = set()
        for data in raw_messages:
            if self.limit and len(records) >= self.limit:
                break
            text, status = _message_text(data, keep_media=self.keep_media)
            if status == "media_skip":
                result.skipped_media += 1
                continue
            if status in ("system", "empty"):
                result.skipped_empty += 1
                continue
            ts = parse_timestamp(str(data.get("time_str") or "")) or _epoch_to_iso(
                data.get("create_time")
            )
            if ts is None:
                result.skipped_time += 1
                continue
            speaker = "me" if data.get("is_sender") else "other"
            key = (ts, speaker, text)
            if key in seen:
                continue
            seen.add(key)
            identifier = str(data.get("server_id") or "").strip()
            records.append(
                MessageRecord(
                    person=person,
                    speaker=speaker,
                    text=text,
                    ts=ts,
                    session=session,
                    source="moect-json",
                    **({"id": f"moect_{identifier}"} if identifier else {}),
                )
            )

        records.sort(key=lambda record: record.ts)
        result.messages = records
        return result
