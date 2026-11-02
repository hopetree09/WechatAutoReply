"""感知层：窗口 → 截图 → OCR → 结构化消息 → 输入框定位。"""

from wagent.perception.types import Box, Message, OcrLine, Speaker

__all__ = ["Box", "Message", "OcrLine", "Speaker"]
