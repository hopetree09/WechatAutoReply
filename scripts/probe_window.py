"""现场诊断脚本：打印微信窗口信息、截图、输入框定位、OCR、气泡配色学习、消息解析。

用法（必须用 wagent 环境的 python）：
    D:\\conda\\envs\\wagent\\python.exe scripts\\probe_window.py
    ... --dump-lines      # 逐行打印 OCR + 分类明细
    ... --save-debug      # 保存标注截图与 OCR JSON 到 screenshots/

说明：输出会对文本做截断，仅用于排查布局问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import get_settings  # noqa: E402
from wagent.perception import regions  # noqa: E402
from wagent.perception.calibration import (  # noqa: E402
    PaletteStore,
    bubble_extent,
    learn_palette,
)
from wagent.perception.capture import WindowCapturer  # noqa: E402
from wagent.perception.locator import (  # noqa: E402
    locate_by_ratio,
    locate_by_uia,
    locate_by_visual,
)
from wagent.perception.ocr import ocr_region  # noqa: E402
from wagent.perception.parser import (  # noqa: E402
    classify_text,
    normalize_text,
    parse_ocr_lines,
    pick_session_name,
)
from wagent.perception.speaker import classify_speaker, estimate_background_rgb  # noqa: E402
from wagent.perception.types import Box  # noqa: E402
from wagent.perception.window import (  # noqa: E402
    WindowNotFoundError,
    find_wechat_window,
    list_windows,
)
from wagent.winapi import enable_dpi_awareness  # noqa: E402


def short(text: str, limit: int = 26) -> str:
    text = normalize_text(text).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-debug", action="store_true", help="保存标注截图")
    parser.add_argument("--dump-lines", action="store_true", help="逐行打印 OCR/分类明细")
    parser.add_argument("--no-calibrate", action="store_true", help="跳过气泡配色自动标定")
    args = parser.parse_args()

    enable_dpi_awareness()
    settings = get_settings()

    print(f"[env] backend={settings.capture.backend} llm={settings.llm.provider}")
    try:
        win = find_wechat_window(settings.wechat)
    except WindowNotFoundError as exc:
        print(f"[!] {exc}")
        print("\n所有可见窗口：")
        for w in list_windows():
            print(f"  [{w.class_name}] {short(w.title, 40)}  {w.width}x{w.height}  proc={w.process_name}")
        return

    print(f"[window] hwnd={win.hwnd} class={win.class_name} proc={win.process_name}")
    print(
        f"[window] title={short(win.title, 40)!r} rect={win.left},{win.top},{win.right},{win.bottom} "
        f"size={win.width}x{win.height}"
    )

    capturer = WindowCapturer(settings.capture)
    img = capturer.grab(win.hwnd)
    h, w = img.shape[:2]
    print(f"[capture] {w}x{h} mean={img.mean():.1f} std={img.std():.1f}")

    panel = regions.chat_panel(w, h, settings.layout)
    title_rect = regions.title_area(w, h, settings.layout)
    print(f"[regions] panel={panel} title={title_rect}")

    # 1) 先定位输入框（三级降级），再用它反推消息区下边界
    uia = locate_by_uia(win.hwnd)
    visual = locate_by_visual(img, settings.layout)
    ratio = locate_by_ratio(img, settings.layout)
    for name, res in (("uia", uia), ("visual", visual), ("ratio", ratio)):
        if res is None:
            print(f"[input:{name}] None")
        else:
            print(f"[input:{name}] box={res.box} conf={res.confidence} click={res.click_point()}")

    msg_rect = regions.message_area(w, h, settings.layout)
    if visual is not None and visual.box.y1 > msg_rect.y1 + 50:
        msg_rect = Box(msg_rect.x1, msg_rect.y1, msg_rect.x2, visual.box.y1 - 6)
    print(f"[regions] message={msg_rect}")

    # 2) 标题区 OCR → 会话名
    title_lines = ocr_region(img, title_rect, settings.ocr)
    session = pick_session_name(title_lines)
    print(f"[session] {session!r}  (title_lines={len(title_lines)})")

    # 3) 消息区 OCR
    msg_lines = ocr_region(img, msg_rect, settings.ocr)
    print(f"[ocr] lines={len(msg_lines)}")

    # 4) 气泡配色：先查缓存，未命中则现场学习并落盘
    background_rgb = estimate_background_rgb(img, msg_rect)
    palette = None
    if settings.calibration.auto and not args.no_calibrate:
        store = PaletteStore(settings.calibration.store_path)
        if background_rgb is not None:
            palette = store.match(background_rgb)
            if palette is not None:
                print(f"[palette] cache hit theme={palette.theme_key} my={palette.my_bgr} other={palette.other_bgr}")
        if palette is None:
            palette = learn_palette(
                img, msg_rect, msg_lines, panel_left=msg_rect.x1, panel_right=msg_rect.x2
            )
            if palette is not None:
                store.upsert(palette)
                print(
                    f"[palette] learned theme={palette.theme_key} bg={palette.background_bgr} "
                    f"my={palette.my_bgr} other={palette.other_bgr}"
                )
    print(f"[palette] background={None if background_rgb is None else background_rgb.round(0).tolist()}")

    my_bgr = palette.my_bgr if palette else None
    other_bgr = palette.other_bgr if palette else None

    def classify(box: Box):
        return classify_speaker(
            img,
            box,
            settings.bubble,
            background_rgb=background_rgb,
            my_bgr=my_bgr,
            other_bgr=other_bgr,
            content_left=msg_rect.x1,
            content_right=msg_rect.x2,
        )

    messages = parse_ocr_lines(
        msg_lines,
        classify,
        session=session,
        extent=lambda box: bubble_extent(img, box),
    )

    if args.dump_lines:
        print(f"[dump] {len(msg_lines)} 行 OCR 明细：")
        for ln in sorted(msg_lines, key=lambda x: (x.box.cy, x.box.cx)):
            spk, conf = classify(ln.box)
            kind = classify_text(ln.text)
            print(
                f"  y={ln.box.cy:>5.0f} x=[{ln.box.x1:>4},{ln.box.x2:>4}] h={ln.box.h:>3} "
                f"s={ln.score:.2f} {spk.value:>7} {conf:.2f} {kind:>9} | {short(ln.text, 30)}"
            )

    print(f"[messages] parsed={len(messages)}")
    for m in messages[-12:]:
        print(f"  {m.speaker.value:>5} y={m.box.cy:>4.0f} conf={m.confidence:.2f} | {short(m.text)}")

    if args.save_debug:
        import cv2

        dbg = img.copy()
        for rect, color in ((panel, (200, 200, 200)), (title_rect, (0, 0, 255)),
                            (msg_rect, (0, 200, 0))):
            cv2.rectangle(dbg, (rect.x1, rect.y1), (rect.x2, rect.y2), color, 1)
        for m in messages:
            color = (0, 0, 255) if m.speaker.value == "me" else (255, 0, 0)
            cv2.rectangle(dbg, (m.box.x1, m.box.y1), (m.box.x2, m.box.y2), color, 1)
        if visual:
            cv2.rectangle(dbg, (visual.box.x1, visual.box.y1), (visual.box.x2, visual.box.y2),
                          (0, 255, 255), 2)
            cv2.circle(dbg, visual.click_point(), 6, (0, 255, 255), -1)

        settings.paths.screenshots_dir.mkdir(parents=True, exist_ok=True)
        out = settings.paths.screenshots_dir / "probe_annotated.png"
        cv2.imwrite(str(out), dbg)
        json_out = settings.paths.screenshots_dir / "probe_ocr.json"
        json_out.write_text(
            json.dumps(
                [
                    {
                        "speaker": m.speaker.value,
                        "text": m.text,
                        "box": [m.box.x1, m.box.y1, m.box.x2, m.box.y2],
                        "confidence": m.confidence,
                    }
                    for m in messages
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[debug] 已保存 {out} 与 {json_out}")


if __name__ == "__main__":
    main()
