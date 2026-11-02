"""标定脚本：探测微信窗口、采样气泡颜色、获取输入框坐标。

用法：
    python scripts/calibrate.py

流程：
    1. 找到微信窗口，打印窗口类名与位置。
    2. 截一张窗口图保存到 screenshots/calibrate.png（供确认）。
    3. 引导你用鼠标依次点击：自己的气泡、对方的气泡、输入框。
    4. 打印采样结果，填入 config.py。

注意：
    - 若屏幕缩放非 100%，颜色采样仍可用；输入框坐标可能有偏差，
      可在 config.py 里手动微调。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 让脚本从 scripts/ 目录运行时也能 import 项目根目录的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pynput import mouse

import config
from core.capture import Capturer


def _list_windows() -> None:
    import win32gui

    def _cb(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            if t:
                print(f"  [{win32gui.GetClassName(h)}] {t}")

    print("当前可见窗口：")
    win32gui.EnumWindows(_cb, None)


def main() -> None:
    import win32gui

    hwnd = win32gui.FindWindow(config.WECHAT_WINDOW_CLASS, None)
    if not hwnd:
        print(f"未找到类名为 {config.WECHAT_WINDOW_CLASS} 的窗口。")
        print("请确认微信已登录。下面列出可见窗口供核对类名：")
        _list_windows()
        return

    rect = win32gui.GetWindowRect(hwnd)
    print(f"[窗口] hwnd={hwnd} rect={rect}")

    cap = Capturer(config.CAPTURE_BACKEND)
    frame = cap.grab_window(hwnd)
    if frame is None:
        print("[!] 截图失败，请尝试把 config.py 的 CAPTURE_BACKEND 改成 mss")
        return

    os.makedirs("screenshots", exist_ok=True)
    cv2.imwrite("screenshots/calibrate.png", frame)
    print("[截图] 已保存到 screenshots/calibrate.png（可打开确认窗口是否截对）")

    print("\n请依次完成 3 次点击（鼠标点进微信窗口内）：")
    print("  第 1 次：点击【你自己的气泡】")
    print("  第 2 次：点击【对方的气泡】")
    print("  第 3 次：点击【输入框】")
    print("每次点击后稍等，程序会记录采样点。\n")

    results: list = []

    def on_click(x, y, button, pressed):
        if not pressed:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        if not (l <= x <= r and t <= y <= b):
            print(f"  (点击在窗口外 ({x},{y})，已忽略)")
            return
        # frame 是物理像素；pynput 坐标在 100% 缩放时等于物理像素
        fx, fy = x - l, y - t
        if 0 <= fx < frame.shape[1] and 0 <= fy < frame.shape[0]:
            px = frame[fy, fx]  # BGR
            rgb = (int(px[2]), int(px[1]), int(px[0]))
        else:
            rgb = (-1, -1, -1)
        results.append((x, y, rgb))
        idx = len(results)
        print(f"  第 {idx} 次采样：屏幕坐标=({x},{y}) 窗口内=({fx},{fy}) RGB={rgb}")
        if len(results) >= 3:
            return False  # 停止监听

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

    print("\n[标定结果] 请填入 config.py：")
    if len(results) >= 1:
        print(f"  MY_BUBBLE_RGB = {results[0][2]}")
    if len(results) >= 2:
        print(f"  OTHER_BUBBLE_RGB = {results[1][2]}")
    if len(results) >= 3:
        print(f"  INPUT_BOX_XY = ({results[2][0]}, {results[2][1]})")


if __name__ == "__main__":
    main()
