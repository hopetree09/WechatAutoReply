"""消息区解析：定位 + OCR + 按气泡颜色分"我/对方"。"""
from __future__ import annotations

import numpy as np

import config
from core.ocr import ocr_lines
from core.state import Message


def find_wechat_hwnd() -> int:
    """定位微信主窗口句柄。"""
    import win32gui

    hwnd = win32gui.FindWindow(config.WECHAT_WINDOW_CLASS, None)
    if not hwnd:
        raise RuntimeError(
            f"未找到微信窗口（类名 {config.WECHAT_WINDOW_CLASS}）。\n"
            "请确认微信已登录且窗口打开；若类名不对，运行 scripts/calibrate.py 探测。"
        )
    return hwnd


def _message_area(img: np.ndarray) -> np.ndarray:
    """按配置比例裁剪出消息区（顶部标题栏、底部输入框之外）。"""
    h = img.shape[0]
    top = int(h * config.MESSAGE_AREA_TOP)
    bottom = int(h * config.MESSAGE_AREA_BOTTOM)
    if bottom <= top:
        return img
    return img[top:bottom, :]


def _classify_speaker(img: np.ndarray, box) -> str:
    """根据气泡背景色判断说话人（近似，需用 calibrate.py 标定颜色）。

    取文本框外接矩形左侧一小块背景像素的平均色，与配置的
    我的/对方气泡颜色做距离比较。
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1 = int(min(xs))
    y1 = int(min(ys))
    y2 = int(max(ys))
    h, w = img.shape[:2]

    # 采样文本框左侧 12px 宽的一小块背景
    sample_x2 = max(0, x1 - 4)
    sample_x1 = max(0, x1 - 16)
    if sample_x2 <= sample_x1:
        return "other"

    roi = img[max(0, y1): min(h, y2), sample_x1:sample_x2]
    if roi.size == 0:
        return "other"

    avg = roi.reshape(-1, 3).mean(axis=0)  # BGR
    my = np.array(config.MY_BUBBLE_RGB[::-1], dtype=float)        # RGB -> BGR
    other = np.array(config.OTHER_BUBBLE_RGB[::-1], dtype=float)
    d_my = float(np.linalg.norm(avg - my))
    d_other = float(np.linalg.norm(avg - other))

    if d_my < config.COLOR_TOLERANCE and d_my < d_other:
        return "me"
    if d_other < config.COLOR_TOLERANCE and d_other < d_my:
        return "other"
    return "me" if d_my < d_other else "other"


def parse_frame(frame) -> list[Message]:
    """把一帧窗口图解析成消息列表（按从上到下的屏幕顺序）。"""
    if frame is None:
        return []

    area = _message_area(frame)
    lines = ocr_lines(area)

    msgs: list[Message] = []
    for text, score, box in lines:
        if not text or score < 0.5:
            continue
        speaker = _classify_speaker(area, box)
        msgs.append(Message(speaker=speaker, text=text.strip()))
    return msgs
