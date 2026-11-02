"""聊天记录导出文件探测：格式识别 + 预览 + 给出一条可直接执行的导入命令。

用户从导出工具拿到的文件可能叫任何名字、可能是 CSV/JSON/JSONL，
先用这个模块看一眼"能不能导、缺什么列、有多少条"，再决定怎么导。
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

Kind = str  # "csv" | "jsonl" | "json" | "unknown"


@dataclass(slots=True)
class ExportPeek:
    path: Path
    kind: Kind
    encoding: str = "utf-8"
    headers: list[str] = field(default_factory=list)
    message_count: int = 0
    sample: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"文件: {self.path}", f"格式: {self.kind}（编码 {self.encoding}）"]
        if self.headers:
            lines.append("表头: " + ", ".join(self.headers))
        lines.append(f"消息数: {self.message_count}")
        for note in self.notes:
            lines.append(f"提示: {note}")
        return "\n".join(lines)

    def suggested_command(self, *, person: str = "") -> str:
        """给出一条可以直接跑的导入命令。"""
        who = person or "<联系人名>"
        if self.kind == "jsonl" or self.kind == "json":
            return (
                f'python -m wagent.import_chat json --file "{self.path}" --person {who}'
            )
        if self.kind == "csv":
            extra = ""
            if self.headers and not any(
                h in self.headers for h in ("是否我发送", "IsSender", "是否发送")
            ):
                extra = '  --me "你的微信昵称"'
            return (
                f'python -m wagent.import_chat csv --file "{self.path}" --person {who}{extra}'
            )
        return "（无法识别格式，请检查文件）"


def _read_head(path: Path, size: int = 8192) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def _decode_guess(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8"


def _jsonl_stats(path: Path, encoding: str) -> tuple[dict, list[dict], int]:
    metadata: dict = {}
    messages: list[dict] = []
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "metadata" in item:
                metadata.update(item["metadata"] or {})
            elif isinstance(item, dict) and "messages" in item:
                metadata.update(item.get("metadata") or {})
                messages.extend(item["messages"])
            elif isinstance(item, dict):
                messages.append(item)
    return metadata, messages, len(messages)


def peek_export(path: Path | str, *, sample_size: int = 3) -> ExportPeek:
    """探测导出文件格式并给出预览。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = _read_head(path)
    text, encoding = _decode_guess(raw)
    peek = ExportPeek(path=path, kind="unknown", encoding=encoding)
    suffix = path.suffix.lower()
    size = path.stat().st_size

    # JSON vs JSONL：
    # - .jsonl 一定按行；.json 优先整体解析（Moect 的 JSON 是单行大 JSON 或美化多行）；
    # - 其余按内容判断：整体能解析就是 JSON，否则试 JSONL。
    if suffix == ".jsonl":
        kind = "jsonl"
        payload = None
    elif suffix == ".json" or (size <= 64 * 1024 * 1024 and text.lstrip("\ufeff").lstrip().startswith(("{", "["))):
        try:
            payload = json.loads(path.read_text(encoding=encoding, errors="replace"))
            kind = "json"
        except json.JSONDecodeError:
            payload = None
            kind = "jsonl"
    else:
        payload = None
        kind = "jsonl" if text.lstrip("\ufeff").lstrip().startswith("{") else "unknown"

    if kind == "json":
        messages = payload.get("messages", []) if isinstance(payload, dict) else payload
        peek.kind = "json"
        peek.message_count = len(messages) if isinstance(messages, list) else 0
        if isinstance(messages, list):
            peek.sample = [item for item in messages[:sample_size] if isinstance(item, dict)]
        peek.notes.append("JSON 导出：结构同 JSONL，字段最全（推荐）")
        return peek

    if kind == "jsonl":
        metadata, messages, count = _jsonl_stats(path, encoding)
        peek.kind = "jsonl"
        peek.message_count = count
        peek.sample = messages[:sample_size]
        if metadata:
            peek.headers = [f"metadata.{key}" for key in list(metadata)[:6]]
        peek.notes.append("Moect/JSONL 导出：秒级时间戳 + is_sender 字段，说话人判定可靠")
        return peek

    # CSV 分支
    reader = csv.reader(io.StringIO(text))
    try:
        headers = next(reader)
    except StopIteration:
        peek.notes.append("空文件")
        return peek
    peek.kind = "csv"
    peek.headers = [header.strip() for header in headers]
    count = 0
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for _line in csv.reader(handle):
            count += 1
    peek.message_count = max(0, count - 1)
    # 采样几行转 dict，方便排错
    with path.open("r", encoding=encoding, errors="replace") as handle:
        reader2 = csv.DictReader(handle)
        peek.sample = [row for _, row in zip(range(sample_size), reader2, strict=False)]
    if not any(h in peek.headers for h in ("时间", "StrTime", "time", "createTime")):
        peek.notes.append("缺少时间列，导入会失败")
    if not any(h in peek.headers for h in ("内容", "StrContent", "content", "text")):
        peek.notes.append("缺少内容列，导入会失败")
    if not any(h in peek.headers for h in ("是否我发送", "IsSender", "是否发送")):
        peek.notes.append("没有『是否我发送』列：需要用 --me 指定你的微信昵称，否则所有消息会被当成对方说的")
    return peek
