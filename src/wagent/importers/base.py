"""导入结果统计（便于 CLI 打印与单测断言）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from wagent.memory.types import MessageRecord


@dataclass(slots=True)
class ImportResult:
    messages: list[MessageRecord] = field(default_factory=list)
    total_rows: int = 0
    skipped_media: int = 0  # 图片/语音/表情等非文本
    skipped_empty: int = 0  # 空内容/系统提示/时间戳
    skipped_time: int = 0  # 时间无法解析
    errors: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def imported(self) -> int:
        return len(self.messages)

    def summary(self) -> str:
        text = (
            f"共 {self.total_rows} 行，导入 {self.imported} 条"
            f"（跳过媒体 {self.skipped_media}、空/系统 {self.skipped_empty}、"
            f"时间异常 {self.skipped_time}、错误 {self.errors}）"
        )
        if self.warnings:
            text += "\n[!] " + "\n[!] ".join(self.warnings)
        return text
