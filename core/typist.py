"""键盘执行：打字速度模型 + 拟人输入 + 发送。"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import pyautogui
import pyperclip

import config


@dataclass
class TypingProfile:
    cps_base: float = 0.30               # 每字基础秒数
    cps_jitter: float = 0.25             # 逐字抖动幅度
    chunk_size: tuple = (12, 22)         # 分片长度（字）
    pause_between_chunks: tuple = (0.8, 2.5)   # 分片间停顿（秒）
    think_delay: tuple = (1.0, 4.0)      # 思考停顿（秒）


def estimate_typing_time(text: str, p: TypingProfile) -> float:
    """估算『假装打字』总时长（秒）。消息越长越久，带随机抖动。"""
    n = len(text)
    if n == 0:
        return 0.0

    total = 0.0
    for _ in text:
        total += p.cps_base * random.uniform(1 - p.cps_jitter, 1 + p.cps_jitter)

    chunk = random.randint(*p.chunk_size)
    pauses = max(0, (n - 1) // chunk)
    total += sum(random.uniform(*p.pause_between_chunks) for _ in range(pauses))
    return total


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def type_humanlike(text: str, p: TypingProfile) -> None:
    """分片输入：emoji/英文逐字打，中文走剪贴板粘贴，片间停顿。

    说明：中文逐字键盘模拟需穿过输入法，工程上不可行，因此中文用
    「剪贴板分片粘贴 + 停顿」近似"一段一段打字"的效果。
    """
    chunk = random.randint(*p.chunk_size)
    pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)]

    for i, piece in enumerate(pieces):
        if _is_ascii(piece):
            pyautogui.write(piece, interval=random.uniform(0.04, 0.12))
        else:
            pyperclip.copy(piece)
            pyautogui.hotkey("ctrl", "v")
        if i < len(pieces) - 1:
            time.sleep(random.uniform(*p.pause_between_chunks))


def send(text: str, p: TypingProfile) -> None:
    """完整发送流程：思考停顿 → 点击输入框 → 输入 → 发送。"""
    time.sleep(random.uniform(*p.think_delay))

    pyautogui.click(*config.INPUT_BOX_XY)   # 点击输入框，确保焦点
    time.sleep(0.3)

    type_humanlike(text, p)
    time.sleep(random.uniform(0.3, 0.8))

    if config.SEND_KEY == "ctrl+enter":
        pyautogui.hotkey("ctrl", "enter")
    else:
        pyautogui.press("enter")
