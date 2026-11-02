"""
配置模板。

使用方式：
    1. 复制本文件为 config.py：  cp config.example.py config.py
    2. 按需修改 config.py 中的参数。
    3. config.py 已被 .gitignore 忽略，不会提交到仓库。

API key 通过环境变量读取（见 .env.example），不要写进本文件。
"""

from __future__ import annotations

import os
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent

# ─────────────────────────── 截图 ───────────────────────────
# 截图后端：dxcam（DDA，推荐，能截自绘窗口）| mss（GDI，可能黑屏）
CAPTURE_BACKEND = "dxcam"
# 轮询新消息的间隔（秒）
POLL_INTERVAL = 1.0

# ─────────────────────────── 微信窗口 ───────────────────────────
# 微信 PC 版主窗口类名（不同版本可能不同，用 scripts/calibrate.py 探测确认）
WECHAT_WINDOW_CLASS = "WeChatMainWndForPC"

# 消息区在窗口中的位置比例（从上到下，0~1），用 scripts/calibrate.py 标定
MESSAGE_AREA_TOP = 0.12       # 消息区顶部（跳过标题栏/会话标题）
MESSAGE_AREA_BOTTOM = 0.72    # 消息区底部（输入框之上）

# ─────────────────────────── 气泡颜色（区分"我/对方"） ───────────────────────────
# 用 scripts/calibrate.py 在你当前主题下采样后填入
MY_BUBBLE_RGB = (149, 236, 105)       # 自己的气泡（默认浅绿）
OTHER_BUBBLE_RGB = (255, 255, 255)    # 对方气泡（默认白色）
COLOR_TOLERANCE = 40                  # 颜色距离容差（0~255，越大越宽松）

# ─────────────────────────── LLM ───────────────────────────
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ─────────────────────────── 打字模拟 ───────────────────────────
TYPING_PARAMS = dict(
    cps_base=0.30,                     # 每字基础秒数（中文聊天约 0.25~0.6）
    cps_jitter=0.25,                   # 逐字抖动幅度（±25%）
    chunk_size=(12, 22),               # 分片长度（字），模拟"打一句停一下"
    pause_between_chunks=(0.8, 2.5),   # 分片间停顿（秒）
    think_delay=(1.0, 4.0),            # 收到消息后"思考"停顿（秒）
)

# 发送键：enter = 回车发送；ctrl+enter = Ctrl+回车发送
SEND_KEY = "enter"

# ─────────────────────────── 风格数据 ───────────────────────────
PROFILE_PATH = ROOT / "data" / "profile.md"      # 你的自述画像
FEWSHOT_DIR = ROOT / "data" / "fewshot"          # 口吻样本目录（*.txt）

# ─────────────────────────── 安全开关 ───────────────────────────
# True：自动点击输入框并发送；False：只打印回复到控制台，不碰键盘鼠标
AUTO_SEND = False

# 输入框中心坐标（屏幕逻辑像素，用 scripts/calibrate.py 标定；仅 AUTO_SEND=True 时使用）
INPUT_BOX_XY = (600, 800)
