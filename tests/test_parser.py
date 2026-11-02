"""解析层单测：不依赖 OCR 引擎，用构造的 OcrLine 覆盖排序/合并/过滤。"""

from __future__ import annotations

from wagent.perception.parser import (
    ParseConfig,
    classify_text,
    normalize_text,
    parse_ocr_lines,
    pick_session_name,
)
from wagent.perception.types import Box, OcrLine, Speaker


def line(text: str, x1: int, y1: int, *, w: int = 80, h: int = 18, score: float = 0.9) -> OcrLine:
    return OcrLine(text=text, score=score, box=Box(x1, y1, x1 + w, y1 + h))


def classify_by_x(box: Box) -> tuple[Speaker, float]:
    """测试桩：左半边算对方，右半边算我。"""
    return (Speaker.OTHER, 0.9) if box.x1 < 300 else (Speaker.ME, 0.9)


# ---------------------------------------------------------------- 文本分类


def test_normalize_text() -> None:
    assert normalize_text("  你 好\u200b  ") == "你 好"
    assert normalize_text("a   b") == "a b"
    assert normalize_text("\u3000行了") == "行了"


def test_classify_timestamp_variants() -> None:
    for text in (
        "21:03",
        "昨天 21:03",
        "2026年10月20日 08:21",
        "10月20日",
        "星期一 21:03",
        "2026-10-20 21:03",
    ):
        assert classify_text(text) == "timestamp", text


def test_classify_system_variants() -> None:
    assert classify_text("对方撤回了一条消息") == "system"
    assert classify_text("我拍了拍「张伟」") == "system"
    assert classify_text("[图片]") == "system"
    assert classify_text("图片") == "system"
    assert classify_text("你已添加了张伟，现在可以开始聊天了") == "system"


def test_classify_message_and_noise() -> None:
    assert classify_text("在吗") == "message"
    assert classify_text("OK 明天见") == "message"
    assert classify_text("") == "noise"
    assert classify_text("123") == "noise"
    assert classify_text("———") == "noise"


# ---------------------------------------------------------------- 排序 / 合并


def test_output_sorted_by_screen_order() -> None:
    lines = [
        line("第三条", 320, 200),
        line("第一条", 320, 100),
        line("第二条", 320, 150),
    ]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert [m.text for m in msgs] == ["第一条", "第二条", "第三条"]


def test_multiline_bubble_merged() -> None:
    # 同一气泡内 3 行，行距 4px（行高 18）
    lines = [
        line("这是被自动换行", 320, 100),
        line("拆成的第二行", 320, 122),
        line("和第三行", 320, 144),
    ]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert len(msgs) == 1
    assert msgs[0].text == "这是被自动换行拆成的第二行和第三行"
    assert msgs[0].speaker is Speaker.ME


def test_ascii_words_keep_space() -> None:
    lines = [line("hello", 320, 100), line("world", 320, 122)]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert msgs[0].text == "hello world"


def test_gap_splits_messages() -> None:
    lines = [line("第一句", 320, 100), line("第二句", 320, 160)]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert [m.text for m in msgs] == ["第一句", "第二句"]


def test_speaker_change_splits_messages() -> None:
    lines = [line("对方说的", 100, 100), line("我说的", 320, 120)]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert [m.speaker for m in msgs] == [Speaker.OTHER, Speaker.ME]


def test_timestamp_breaks_merge() -> None:
    lines = [
        line("上一句", 320, 100),
        line("昨天 21:03", 250, 120, w=60),
        line("下一句", 320, 140),
    ]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert [m.text for m in msgs] == ["上一句", "下一句"]


def test_unknown_speaker_dropped() -> None:
    def always_unknown(_box: Box) -> tuple[Speaker, float]:
        return Speaker.UNKNOWN, 0.0

    msgs = parse_ocr_lines([line("无法判定", 320, 100)], always_unknown)
    assert msgs == []


def test_low_score_and_short_lines_filtered() -> None:
    lines = [
        line("低置信度", 320, 100, score=0.3),
        OcrLine(text="细线", score=0.99, box=Box(320, 200, 380, 203)),
    ]
    msgs = parse_ocr_lines(lines, classify_by_x, cfg=ParseConfig(min_line_height=8))
    assert msgs == []


def test_x_offset_prevents_merge() -> None:
    lines = [line("左", 320, 100), line("右", 480, 120)]
    msgs = parse_ocr_lines(lines, classify_by_x)
    assert len(msgs) == 2


def test_extent_groups_multiline_and_separates_bubbles() -> None:
    """同一气泡（左右边界一致）内的行要合并；不同气泡即使行距小也不能合并。"""
    lines = [
        line("卡片第一行", 320, 100),
        line("卡片第二行", 320, 122),
        line("另一个气泡", 320, 144),
    ]

    def extent(box: Box) -> tuple[int, int]:
        if box.y1 <= 122:
            return (300, 800)  # 前两行同属一个宽气泡
        return (600, 800)  # 第三个是另一个气泡

    msgs = parse_ocr_lines(lines, classify_by_x, extent=extent)
    assert [m.text for m in msgs] == ["卡片第一行卡片第二行", "另一个气泡"]


def test_extent_allows_slightly_larger_gap() -> None:
    lines = [line("第一行", 320, 100), line("第二行", 320, 130)]  # gap=12, 行高18

    def extent(_box: Box) -> tuple[int, int]:
        return (300, 800)

    msgs = parse_ocr_lines(lines, classify_by_x, extent=extent)
    assert len(msgs) == 1


def test_fingerprint_changes_by_position_and_speaker() -> None:
    a = parse_ocr_lines([line("哈哈", 320, 100)], classify_by_x)[0]
    b = parse_ocr_lines([line("哈哈", 320, 300)], classify_by_x)[0]
    c = parse_ocr_lines([line("哈哈", 100, 100)], classify_by_x)[0]
    assert a.fingerprint != b.fingerprint  # 同一句话在不同位置 = 不同消息
    assert a.fingerprint != c.fingerprint  # 说我 vs 对方说


def test_confidence_propagated() -> None:
    def conf_04(box: Box) -> tuple[Speaker, float]:
        return Speaker.OTHER, 0.4

    msg = parse_ocr_lines([line("嗯", 100, 100)], conf_04)[0]
    assert msg.confidence == 0.4


# ---------------------------------------------------------------- 会话名


def test_pick_session_name() -> None:
    lines = [
        line("昨天 21:03", 250, 20, w=60),
        line("路旁的树", 300, 40, w=70),
        line("在吗", 320, 120),
    ]
    assert pick_session_name(lines) == "路旁的树"


def test_pick_session_name_empty() -> None:
    assert pick_session_name([line("21:03", 250, 20)]) == ""
