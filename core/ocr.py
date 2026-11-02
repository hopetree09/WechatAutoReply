"""OCR 封装：RapidOCR 离线识别。"""
from __future__ import annotations

from typing import Optional

_engine = None


def get_engine():
    """懒加载 RapidOCR 引擎（首次调用会加载模型，较慢）。"""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def ocr_lines(img) -> list:
    """对图像做 OCR。

    参数：
        img: numpy BGR 数组 (H, W, 3)，或 None。

    返回：
        [(text, score, box), ...]
          - text:  识别文本
          - score: 置信度 0~1
          - box:   四点坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]（左上起顺时针）
    """
    if img is None:
        return []

    result, _ = get_engine()(img)
    if not result:
        return []

    out = []
    for box, text, score in result:
        out.append((text, float(score), box))
    return out
