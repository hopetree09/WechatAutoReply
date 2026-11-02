"""配置与区域划分单测。"""

from __future__ import annotations

from wagent.config import LayoutSettings, get_settings, reset_settings
from wagent.perception.regions import chat_panel, input_area, message_area, title_area
from wagent.perception.types import Box


def test_defaults_capture_backend_prefers_printwindow() -> None:
    reset_settings()
    s = get_settings()
    # 微信 4.x 是 Qt 窗口，PrintWindow 能截被遮挡的窗口
    assert s.capture.backend == "printwindow"
    assert s.llm.provider == "local"


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("WAGENT_LLM__PROVIDER", "cloud")
    monkeypatch.setenv("WAGENT_CAPTURE__BACKEND", "dxcam")
    reset_settings()
    s = get_settings()
    assert s.llm.provider == "cloud"
    assert s.capture.backend == "dxcam"
    reset_settings()


def test_regions_are_nested() -> None:
    layout = LayoutSettings()
    w, h = 1000, 800
    panel = chat_panel(w, h, layout)
    title = title_area(w, h, layout)
    msgs = message_area(w, h, layout)
    inputs = input_area(w, h, layout)

    assert panel.x1 == int(w * layout.chat_panel_left)
    assert title.y2 <= msgs.y1
    assert msgs.y2 <= inputs.y1
    assert all(r.x1 >= panel.x1 and r.x2 <= panel.x2 for r in (title, msgs, inputs))


def test_box_helpers() -> None:
    box = Box(10, 20, 110, 60)
    assert box.w == 100 and box.h == 40
    assert box.cx == 60 and box.cy == 40
    assert box.clip(50, 50) == Box(10, 20, 50, 50)
    assert box.expand(2, 3) == Box(8, 17, 112, 63)
    assert Box.from_quad([[10, 20], [110, 20], [110, 60], [10, 60]]) == box
