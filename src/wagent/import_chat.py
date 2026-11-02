"""导入聊天记录到本地记忆库。

两种数据来源：

    1) CSV（先用你信任的工具导出，仓库本身不含解密代码）
       python -m wagent.import_chat csv --file chat.csv --person 张三

    2) 滚动 OCR 采史（不依赖任何导出工具，直接滚动微信窗口）
       python -m wagent.import_chat scroll --person 张三 --max-frames 20

通用参数：
       --db data/agent.db   指定数据库
       --dry-run            只统计不写库
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from wagent.config import get_settings
from wagent.importers import CsvImporter, MoectJsonImporter, ScrollHarvester, peek_export
from wagent.memory.store import MemoryStore


def import_csv(args: argparse.Namespace, store: MemoryStore) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"[!] 文件不存在：{path}")
        return 2
    importer = CsvImporter(
        path=path,
        person=args.person,
        me_names=tuple(args.me),
        encoding=args.encoding,
    )
    result = importer.run()
    print(f"[csv] {path.name}: {result.summary()}")
    if result.messages:
        print(f"[csv] 时间范围：{result.messages[0].ts} → {result.messages[-1].ts}")
    if not args.dry_run:
        added = store.add_messages(result.messages)
        print(f"[db] 新增 {added} 条（已按指纹去重），当前：{store.stats()}")
    return 0


def import_scroll(args: argparse.Namespace, store: MemoryStore) -> int:
    from wagent.im.wechat import WeChatAdapter
    from wagent.winapi import enable_dpi_awareness, move_mouse, scroll

    enable_dpi_awareness()
    settings = get_settings()
    try:
        adapter = WeChatAdapter(settings)
    except Exception as exc:
        print(f"[!] 连接微信失败：{exc}")
        return 2

    first = adapter.capture()
    session = first.session
    person = args.person or session
    if not person:
        print("[!] 读不到会话名，请用 --person 指定联系人")
        return 2
    rect = adapter.message_rect
    if rect is None:
        print("[!] 消息区定位失败")
        return 2
    center = (rect.x1 + rect.w // 2, rect.y1 + rect.h // 2)
    print(f"[scroll] 会话={session!r} 联系人={person} 消息区={rect} 中心={center}")
    print("[scroll] 开始向上滚动采史（窗口会动，请先不要操作微信）...")

    def grab():
        return adapter.capture().messages

    def scroll_up() -> None:
        move_mouse(*center, settle=0.05)
        scroll(3)

    harvester = ScrollHarvester(
        grab_messages=grab,
        scroll_up=scroll_up,
        person=person,
        session=session,
        max_frames=args.max_frames,
        max_stagnant=2,
        settle=lambda: time.sleep(args.settle),
        now=lambda: __import__("datetime").datetime.now().astimezone(),
    )
    result = harvester.harvest()
    print(f"[scroll] {result.summary()}")
    if result.messages:
        print(f"[scroll] 时间范围：{result.messages[0].ts} → {result.messages[-1].ts}")
    if not args.dry_run:
        added = store.add_messages(result.messages)
        print(f"[db] 新增 {added} 条（已按指纹去重），当前：{store.stats()}")
    return 0


def import_json(args: argparse.Namespace, store: MemoryStore) -> int:
    """导入 Moect 的 JSON/JSONL 导出（字段最全，能确定区分我/对方）。"""
    path = Path(args.file)
    if not path.exists():
        print(f"[!] 文件不存在：{path}")
        return 2
    importer = MoectJsonImporter(
        path=path,
        person=args.person,
        session=args.session,
        keep_media=args.keep_media,
    )
    try:
        result = importer.run()
    except Exception as exc:
        print(f"[!] 解析失败：{exc}")
        return 2
    print(f"[json] {path.name}: {result.summary()}")
    if result.messages:
        print(f"[json] 时间范围：{result.messages[0].ts} → {result.messages[-1].ts}")
        if not args.dry_run:
            added = store.add_messages(result.messages)
            print(f"[db] 新增 {added} 条（已按指纹去重），当前：{store.stats()}")
    return 0


def inspect_file(args: argparse.Namespace, _store: MemoryStore) -> int:
    """看一眼导出的文件是什么格式、能不能导、该用什么命令导。"""
    try:
        peek = peek_export(Path(args.file))
    except FileNotFoundError:
        print(f"[!] 文件不存在：{args.file}")
        return 2
    print(peek.summary())
    if peek.sample:
        print("样例：")
        for row in peek.sample[:2]:
            print("  ", str(row)[:300])
    print("\n建议命令：")
    print(peek.suggested_command(person=args.person))
    return 0

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入微信聊天记录到本地记忆库")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--person", default="", help="联系人/会话名")
    common.add_argument("--db", default="", help="SQLite 数据库路径（默认 data/agent.db）")
    common.add_argument("--dry-run", action="store_true", help="只统计不写库")

    csv_parser = sub.add_parser("csv", parents=[common], help="从 CSV 导入")
    csv_parser.add_argument("--file", required=True, help="CSV 文件路径")
    csv_parser.add_argument("--encoding", default=None, help="强制编码（默认自动识别）")
    csv_parser.add_argument(
        "--me", nargs="*", default=["我", "me", "self"], help="判断「我」的发送人名字"
    )

    json_parser = sub.add_parser("json", parents=[common], help="从 Moect JSON/JSONL 导入（推荐）")
    json_parser.add_argument("--file", required=True, help=".json 或 .jsonl 文件路径")
    json_parser.add_argument("--session", default="", help="会话名（默认取导出元数据里的名字）")
    json_parser.add_argument("--keep-media", action="store_true", help="图片/表情等保留为 [类型] 占位")

    inspect_parser = sub.add_parser("inspect", parents=[common], help="探测导出文件格式并给出导入命令")
    inspect_parser.add_argument("--file", required=True)

    scroll_parser = sub.add_parser("scroll", parents=[common], help="滚动 OCR 采史")
    scroll_parser.add_argument("--max-frames", type=int, default=20)
    scroll_parser.add_argument("--settle", type=float, default=0.8, help="滚动后等待秒数")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    db_path = Path(args.db) if args.db else settings.paths.db_path
    store = MemoryStore(db_path)
    print(f"[db] {db_path}")
    try:
        if args.command == "csv":
            return import_csv(args, store)
        if args.command == "json":
            return import_json(args, store)
        if args.command == "inspect":
            return inspect_file(args, store)
        return import_scroll(args, store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
