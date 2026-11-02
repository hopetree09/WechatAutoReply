"""离线聊天记录导入：CSV / Moect JSON(L) / 滚动 OCR 采史。

三种数据来源：
- :mod:`wagent.importers.csv_importer`：WeChatMsg / Moect / 手工导出的 CSV；
- :mod:`wagent.importers.moect_json`：Moect 的 JSON/JSONL（字段最全，推荐）；
- :mod:`wagent.importers.scroll_harvester`：不依赖任何导出工具，滚动 OCR 采史。

仓库本身**不含任何解密代码**，数据来源由使用者自行负责。
"""

from wagent.importers.base import ImportResult
from wagent.importers.csv_importer import ColumnMapping, CsvImporter
from wagent.importers.detect import ExportPeek, peek_export
from wagent.importers.moect_json import MoectJsonImporter
from wagent.importers.scroll_harvester import HarvestResult, ScrollHarvester, merge_frames

__all__ = [
    "ColumnMapping",
    "CsvImporter",
    "ExportPeek",
    "HarvestResult",
    "ImportResult",
    "MoectJsonImporter",
    "ScrollHarvester",
    "merge_frames",
    "peek_export",
]
