"""CLI 入口的端到端测试：离线 mock 演示必须能跑通且默认不发送。"""

from __future__ import annotations

from wagent.main import main


def test_mock_demo_dry_run(capsys) -> None:
    code = main(["--mock", "--fake-llm", "--interval", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "backend=mock" in out
    assert "mode=dry_run" in out
    # 基线帧不回；随后一条对方新消息产生预览；自己的回复出现后不再回
    assert out.count("预览·未发送") == 2
    assert "无新消息" in out
    assert "已发送" not in out


def test_mock_demo_send_mode_blocked_by_foreground_guard(capsys) -> None:
    """mock 场景里 hwnd=1，真实前台窗口不可能是它 → 发送必须被前台校验拦下。"""
    code = main(["--mock", "--fake-llm", "--mode", "send", "--interval", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "not_foreground" in out
    assert "已发送" not in out


def test_bad_mode_rejected(capsys) -> None:
    try:
        main(["--mock", "--mode", "nope"])
    except SystemExit as exc:  # argparse 对非法参数会退出
        assert exc.code == 2
    else:  # pragma: no cover - argparse 行为变化时的兜底
        raise AssertionError("非法模式应当被 argparse 拒绝")
