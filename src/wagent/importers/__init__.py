"""离线聊天记录导入：CSV 文件 / 滚动 OCR 采史。

两种数据来源：
- :mod:`wagent.importers.csv_importer`：把 WeChatMsg / PyWxDump / 手工导出的 CSV
  转成统一的 ``MessageRecord``——仓库本身**不含任何解密代码**，数据来源由使用者自行负责；
- :mod:`wagent.importers.scroll_harvester`：不依赖任何导出工具，直接滚动微信窗口
  用 OCR 把历史"看"下来（慢、但零风险，且完全在本项目的能力边界内）。
"""

from wagent.importers.base import ImportResult
from wagent.importers.csv_importer import ColumnMapping, CsvImporter
from wagent.importers.scroll_harvester import HarvestResult, ScrollHarvester, merge_frames

__all__ = [
    "ColumnMapping",
    "CsvImporter",
    "HarvestResult",
    "ImportResult",
    "ScrollHarvester",
    "merge_frames",
]
