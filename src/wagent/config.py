"""全局配置：pydantic-settings，支持 .env 与环境变量覆盖。

约定：
- 环境变量前缀 ``WAGENT_``，嵌套项用 ``__`` 连接，例如
  ``WAGENT_LLM__PROVIDER=local``、``WAGENT_BUBBLE__MY_RGB="149,236,105"``。
- 本地密钥放在项目根 ``.env``（已在 .gitignore 中）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent


class CaptureSettings(BaseModel):
    """截图后端。

    微信 4.x 是 Qt 窗口，``printwindow``（PW_RENDERFULLCONTENT）可以在窗口被遮挡、
    甚至不在前台时截到内容；``dxcam`` 只能截可见屏幕；``mss`` 是 GDI 兜底。
    """

    backend: Literal["printwindow", "dxcam", "mss"] = "printwindow"
    # dxcam 截屏时是否把窗口临时置顶（仅 backend=dxcam 有效）
    raise_window: bool = False


class WeChatSettings(BaseModel):
    """微信窗口识别规则。"""

    # 微信 4.x 主窗口类名形如 Qt51514QWindowIcon（随 Qt 版本变化，用正则匹配）
    window_class_regex: str = r"^(Qt\d+QWindowIcon|WeChatMainWndForPC|Chrome_WidgetWin_0)$"
    title_keywords: tuple[str, ...] = ("微信", "Weixin", "WeChat")
    process_names: tuple[str, ...] = ("Weixin.exe", "WeChat.exe")
    # 主窗口被最小化时是否自动还原（最小化窗口截不到内容）
    restore_minimized: bool = True


class LayoutSettings(BaseModel):
    """窗口内布局比例（按窗口高度/宽度归一化）与消息分组参数。"""

    # 聊天标题栏高度比例（标题文字所在条带）
    title_top: float = 0.03
    # 消息区（顶部标题栏之下、输入框之上）占窗口高度的比例；
    # message_area_bottom 只是「输入框搜索」的起点提示，实际下边界由输入框定位结果决定
    message_area_top: float = 0.11
    message_area_bottom: float = 0.72
    # 左侧会话列表占窗口宽度的比例（用于把消息区限制在右侧聊天面板）
    chat_panel_left: float = 0.41
    # 同一气泡内多行合并：行间垂直间隙 <= ratio * 行高 视为同一条消息
    line_merge_gap_ratio: float = 0.6
    # 文本行最短高度（过滤水印/横线类噪声）
    min_line_height: int = 8


class BubbleSettings(BaseModel):
    """说话人判别：气泡底色投票。

    判定用「三分类最近邻」：每个采样块归到 {我, 对方, 背景} 中色距最近的类，
    再按权重投票。背景色不用手工配置——运行时从消息区统计得到（``estimate_background_rgb``），
    这样浅灰背景不会被误判成对方的白气泡。
    """

    my_rgb: tuple[int, int, int] = (149, 236, 105)
    other_rgb: tuple[int, int, int] = (255, 255, 255)
    # 采样块与已知类别（我/对方/背景）的最大色距，超过则这一票作废
    tolerance: float = 40.0
    # 判定所需的最低票数比例（相对全部有效票）
    min_vote_ratio: float = 0.5
    # 采样块最小边长（像素），太小则跳过
    min_patch_px: int = 3


class CalibrationSettings(BaseModel):
    """气泡配色自动标定。"""

    auto: bool = True
    store_path: Path = ROOT / "data" / "calibration.json"


class OcrSettings(BaseModel):
    min_score: float = 0.5
    # 送 OCR 前把截图放大，提升小字号识别率（1.0 表示不缩放）
    upscale: float = 1.0


class LlmSettings(BaseModel):
    """LLM 双通道：默认本地（聊天内容不出本机），可切云端。"""

    provider: Literal["local", "cloud"] = "local"

    # 本地：Ollama / LM Studio / vLLM 等 OpenAI 兼容服务
    local_base_url: str = "http://127.0.0.1:11434/v1"
    local_model: str = "qwen2.5:7b-instruct-q4_K_M"
    local_api_key: str = "local"

    # 云端：默认 DeepSeek
    cloud_base_url: str = "https://api.deepseek.com/v1"
    cloud_model: str = "deepseek-chat"
    cloud_api_key: str = ""

    temperature: float = 0.9
    max_tokens: int = 300
    timeout_s: float = 60.0
    max_retries: int = 2


class TypingSettings(BaseModel):
    """拟人化打字节奏（详见 wagent.actuator.TypingProfile）。"""

    cps_base: float = 0.30
    cps_jitter: float = 0.25
    chunk_size: tuple[int, int] = (12, 22)
    pause_between_chunks: tuple[float, float] = (0.8, 2.5)
    think_delay: tuple[float, float] = (1.0, 4.0)
    send_key: Literal["enter", "ctrl+enter"] = "enter"


class AgentSettings(BaseModel):
    """主循环/安全策略。"""

    # 默认只出计划不发送；review 需要人工确认；send 才真正发
    mode: Literal["dry_run", "review", "send"] = "dry_run"
    poll_interval_s: float = 1.0
    # 发送前要求微信在前台（防止把回复打进别的程序）
    require_foreground: bool = True


class PathsSettings(BaseModel):
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    screenshots_dir: Path = ROOT / "screenshots"
    db_path: Path = ROOT / "data" / "agent.db"
    profile_path: Path = ROOT / "data" / "profile.md"
    fewshot_dir: Path = ROOT / "data" / "fewshot"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WAGENT_",
        env_nested_delimiter="__",
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    wechat: WeChatSettings = Field(default_factory=WeChatSettings)
    layout: LayoutSettings = Field(default_factory=LayoutSettings)
    bubble: BubbleSettings = Field(default_factory=BubbleSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    typing: TypingSettings = Field(default_factory=TypingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)


_settings: Settings | None = None


def get_settings() -> Settings:
    """进程内单例，避免重复读 .env。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """测试用：清掉缓存，让下一次 get_settings() 重新读取环境变量。"""
    global _settings
    _settings = None
