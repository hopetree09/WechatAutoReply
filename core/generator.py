"""LLM 生成：带个人风格的回复。"""
from __future__ import annotations

import re

import config


def _load_profile() -> str:
    if config.PROFILE_PATH.exists():
        return config.PROFILE_PATH.read_text(encoding="utf-8").strip()
    return "（未提供自述画像，请创建 data/profile.md）"


def _load_fewshot() -> str:
    if not config.FEWSHOT_DIR.exists():
        return ""
    lines: list[str] = []
    for fp in sorted(config.FEWSHOT_DIR.glob("*.txt")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    return "\n".join(lines[:200])


def _system_prompt() -> str:
    profile = _load_profile()
    fewshot = _load_fewshot()
    fewshot_block = fewshot if fewshot else "（暂无样本）"
    return f"""你是「我」本人，正在微信里和熟人聊天。完全代入我的身份、经历、语气和表情习惯回复。

【我是什么样的人】
{profile}

【我的说话风格，必须严格遵守】
- 用短句，句尾通常不加句号，偶尔用「～」结尾。
- 不说「首先/其次/总之」，不总结不复述，不解释自己为什么这么回。
- 不用「亲/您/加油哦」这类客套，不排比、不凑三段式。
- 允许不完整的句子和口头语。
- 适当用我常用的 emoji，但别每条都塞。

【我最近真实说过的话（照此口吻写）】
{fewshot_block}

【回复要求】
- 直接回答对方，像真人随手打字，不要 AI 味。
- 长度自然：对方问一句就回一句，别长篇大论。"""


def _clean(raw: str) -> str:
    """出口清洗：剥编号/符号前缀/引号/句尾句号。"""
    text = raw.strip()
    text = re.sub(r"^[>\-*#\d\.\s]+", "", text)
    text = text.strip("\"'`[]（）()")
    if text.endswith("。") and not text.endswith(("？！～")):
        text = text[:-1]
    return text


class Generator:
    def __init__(self) -> None:
        if not config.LLM_API_KEY:
            raise RuntimeError(
                "缺少 LLM_API_KEY 环境变量。请先设置（参考 .env.example）：\n"
                "  PowerShell:  $env:LLM_API_KEY = \"你的key\""
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        self.model = config.LLM_MODEL
        self.system = _system_prompt()

    def reply(self, context: list[str], latest: str) -> str:
        history = "\n".join(context[-12:])
        user = (
            "最近聊天记录：\n"
            f"{history}\n\n"
            f"对方刚发来：{latest}\n\n"
            "请以我的口吻回复（只输出回复内容，不要任何前缀、编号、引号）："
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            temperature=0.9,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content or ""
        return _clean(raw)
