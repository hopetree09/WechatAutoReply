"""帧签名/缓存单测。"""

from __future__ import annotations

import numpy as np

from wagent.perception.framesig import FrameCache, frame_signature
from wagent.perception.types import Box


def make_image(value: int = 240, size: tuple[int, int] = (200, 300)) -> np.ndarray:
    return np.full((*size, 3), value, dtype=np.uint8)


def test_signature_stable_for_same_image() -> None:
    img = make_image()
    region = Box(0, 0, 300, 200)
    assert frame_signature(img, region) == frame_signature(img, region)


def test_signature_changes_when_content_changes() -> None:
    img = make_image()
    region = Box(0, 0, 300, 200)
    before = frame_signature(img, region)
    img2 = img.copy()
    img2[50:100, 50:100] = 20
    assert frame_signature(img2, region) != before


def test_signature_ignores_tiny_render_jitter() -> None:
    """量化到 4 级后，±1 的渲染抖动不应导致签名变化。"""
    img = make_image(240)
    region = Box(0, 0, 300, 200)
    img2 = img.copy()
    img2[:, :] = 241
    assert frame_signature(img2, region) == frame_signature(img, region)


def test_cache_hits_when_unchanged() -> None:
    img = make_image()
    region = Box(0, 0, 300, 200)
    cache: FrameCache[str] = FrameCache()
    calls = {"n": 0}

    def compute() -> str:
        calls["n"] += 1
        return "parsed"

    value, hit = cache.get_or_compute(img, region, compute)
    assert value == "parsed" and hit is False
    value, hit = cache.get_or_compute(img.copy(), region, compute)
    assert hit is True and calls["n"] == 1
    assert cache.hits == 1 and cache.misses == 1


def test_cache_miss_when_content_changes() -> None:
    img = make_image()
    region = Box(0, 0, 300, 200)
    cache: FrameCache[str] = FrameCache()
    calls = {"n": 0}

    def compute() -> str:
        calls["n"] += 1
        return f"parsed-{calls['n']}"

    cache.get_or_compute(img, region, compute)
    changed = img.copy()
    changed[10:60, 10:60] = 0
    value, hit = cache.get_or_compute(changed, region, compute)
    assert hit is False
    assert value == "parsed-2"


def test_invalidate_forces_recompute() -> None:
    img = make_image()
    region = Box(0, 0, 300, 200)
    cache: FrameCache[str] = FrameCache()
    cache.get_or_compute(img, region, lambda: "a")
    cache.invalidate()
    value, hit = cache.get_or_compute(img, region, lambda: "b")
    assert hit is False and value == "b"
