"""主循环：感知 → 生成 → 执行。

运行前请完成：
    1. pip install -r requirements.txt
    2. cp config.example.py config.py
    3. 设置环境变量 LLM_API_KEY（见 .env.example）
    4. python scripts/calibrate.py  标定窗口与气泡颜色
    5. python main.py
"""
from __future__ import annotations

import time

import config
from core.capture import Capturer
from core.generator import Generator
from core.parser import find_wechat_hwnd, parse_frame
from core.state import Conversation
from core.typist import TypingProfile, send


def main() -> None:
    print("[*] 初始化...")
    hwnd = find_wechat_hwnd()
    capturer = Capturer(backend=config.CAPTURE_BACKEND)
    gen = Generator()
    profile = TypingProfile(**config.TYPING_PARAMS)
    conv = Conversation()

    print(f"[*] 已连接微信窗口 (hwnd={hwnd})，开始监听...")
    print(f"[*] 自动发送：{'开' if config.AUTO_SEND else '关（只打印，不发送）'}")
    print("[*] 按 Ctrl+C 退出\n")

    while True:
        try:
            frame = capturer.grab_window(hwnd)
            msgs = parse_frame(frame)

            # 屏幕最下方一条是"对方"消息，且是新增的，才触发回复
            if msgs and msgs[-1].speaker == "other" and conv.update(msgs):
                latest = msgs[-1].text
                print(f"[对方] {latest}")

                reply = gen.reply(conv.context_texts(), latest)
                print(f"[回复] {reply}")

                if config.AUTO_SEND:
                    send(reply, profile)
                    print("[已发送]")

            time.sleep(config.POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            print(f"[!] {e}")
            time.sleep(config.POLL_INTERVAL)

    print("\n[*] 已退出")


if __name__ == "__main__":
    main()
