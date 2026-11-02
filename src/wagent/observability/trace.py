"""结构化 trace：每一步的耗时/动作/错误。

目标：出问题时能回答「这一轮到底卡在哪」——感知慢？模型慢？发送被拦？
trace 同时写到内存（测试）和 JSONL 文件（事后分析）。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class TraceEvent:
    seq: int
    ts: str
    kind: str
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"seq": self.seq, "ts": self.ts, "kind": self.kind, **self.data}


class Tracer:
    """轻量 tracer：事件列表 + 可选 JSONL 落盘。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.events: list[TraceEvent] = []
        self.path: Path | None = Path(path) if path else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

    def event(self, kind: str, **data: object) -> TraceEvent:
        event = TraceEvent(
            seq=len(self.events) + 1,
            ts=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            kind=kind,
            data=dict(data),
        )
        self.events.append(event)
        if self._handle is not None:
            self._handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
            self._handle.flush()
        return event

    def span(self, kind: str, started: float, **data: object) -> TraceEvent:
        """记录一个已完成的 span（自动带上毫秒耗时）。"""
        return self.event(kind, ms=round((time.perf_counter() - started) * 1000, 1), **data)

    def summary(self) -> dict:
        counts = Counter(event.kind for event in self.events)
        totals: dict[str, float] = {}
        for event in self.events:
            ms = event.data.get("ms")
            if isinstance(ms, (int, float)):
                totals[event.kind] = round(totals.get(event.kind, 0.0) + float(ms), 1)
        errors = sum(1 for event in self.events if event.kind == "error")
        return {
            "events": len(self.events),
            "counts": dict(counts),
            "total_ms": totals,
            "errors": errors,
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
