"""RapidOCR 封装（离线）。

- 引擎懒加载：首次调用会下载/加载 ONNX 模型（需联网）。
- 坐标一律换算回**整窗截图**坐标系，方便上层按窗口比例算布局。
"""

from __future__ import annotations

import numpy as np

from wagent.config import OcrSettings
from wagent.perception.types import Box, OcrLine

_engine = None


def get_engine():
    """懒加载 RapidOCR 单例。"""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def ocr_image(img: np.ndarray, settings: OcrSettings | None = None) -> list[OcrLine]:
    """对 BGR 图像做 OCR，返回按原始坐标系的结果。"""
    if img is None or img.size == 0:
        return []
    settings = settings or OcrSettings()

    scale = settings.upscale
    if scale and abs(scale - 1.0) > 1e-3:
        import cv2

        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    result, _ = get_engine()(img)
    if not result:
        return []

    lines: list[OcrLine] = []
    for quad, text, score in result:
        text = (text or "").strip()
        score = float(score)
        if not text or score < settings.min_score:
            continue
        box = Box.from_quad(quad)
        if scale and abs(scale - 1.0) > 1e-3:
            inv = 1.0 / scale
            box = Box(
                int(box.x1 * inv),
                int(box.y1 * inv),
                int(box.x2 * inv),
                int(box.y2 * inv),
            )
        lines.append(OcrLine(text=text, score=score, box=box))
    return lines


def ocr_region(
    img: np.ndarray, region: Box, settings: OcrSettings | None = None
) -> list[OcrLine]:
    """只 OCR 指定区域，并把结果坐标平移回整窗坐标系。"""
    h, w = img.shape[:2]
    region = region.clip(w, h)
    if region.w <= 0 or region.h <= 0:
        return []
    sub = img[region.y1 : region.y2, region.x1 : region.x2]
    lines = ocr_image(sub, settings)
    return [
        OcrLine(
            text=ln.text,
            score=ln.score,
            box=Box(
                ln.box.x1 + region.x1,
                ln.box.y1 + region.y1,
                ln.box.x2 + region.x1,
                ln.box.y2 + region.y1,
            ),
        )
        for ln in lines
    ]
