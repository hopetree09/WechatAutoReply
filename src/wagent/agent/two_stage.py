"""两段式生成：7B（或云端）出「内容草稿」→ 3B LoRA 改写成「我的口吻」。

为什么这样分工：
- 7B 能读长 prompt（画像 + 记忆），负责"说什么"（内容/事实/情绪）；
- 3B LoRA 只在短 prompt 分布下训练过，负责"怎么说"（极短、口语、像本人）。

用法（脚本里）：``TwoStageGenerator(...)(trigger)``。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from wagent.agent.state import Trigger
from wagent.llm.client import ChatModel
from wagent.llm.prompt import (
    build_messages,
    clean_reply,
    render_context,
)

# 改写阶段用短 system prompt（贴近 3B LoRA 训练时的分布），不要塞画像/记忆
REWRITE_SYSTEM = (
    "你是我的微信回复改写器。把「草稿」改写成我平时发微信的口吻："
    "极短、口语、几乎不用标点；保留草稿的意思和关键信息；"
    "不要添加新事实，不要解释，不要带引号。只输出改写后的那一句。"
)


REWRITE_FEWSHOT = (
    ("可以啊，你定地方就行，我这边都可以。", "行 你定"),
    ("我觉得不用太担心，应该没什么大问题的。", "问题不大"),
    ("下周我得回学校一趟，周日下午到。", "下周回学校 周日下午到"),
)


def build_rewrite_messages(
    context: Sequence,
    latest: str,
    draft: str,
    *,
    fewshot: bool = True,
) -> list:
    """构造 3B 改写阶段的 messages（短 system + 上下文 + 草稿 + 改写示例）。

    3B 从没在「改写」任务上训过，只给一句"请改写"它会直接当聊天回复；
    给 3 个（草稿→改写）示例能把它拉回"转换"模式。
    """
    from wagent.llm.client import MessageTurn

    history = render_context(context, limit=12)
    examples = ""
    if fewshot:
        examples = "改写示例：\n" + "\n".join(
            f"草稿：{before}\n改写：{after}" for before, after in REWRITE_FEWSHOT
        ) + "\n\n"
    user = (
        f"最近聊天记录：\n{history}\n\n"
        f"对方刚发来：{latest}\n\n"
        f"{examples}"
        f"草稿：{draft}\n\n"
        "请以我的口吻改写（保留草稿的意思，只输出回复内容）："
    )
    return [MessageTurn("system", REWRITE_SYSTEM), MessageTurn("user", user)]


@dataclass
class TwoStageGenerator:
    """第一段草稿（7B/云端）+ 第二段风格改写（3B LoRA）。"""

    draft_llm: ChatModel
    rewrite_llm: ChatModel
    profile: str = ""
    fewshot: Sequence[str] = ()
    limit: int = 12
    memory_provider: Callable[[Trigger], str] | None = None
    memory_policy: Callable[[str], bool] | None = None
    on_draft: Callable[[str], None] | None = field(default=None, repr=False)

    def _memory_block(self, trigger: Trigger) -> str:
        if self.memory_provider is None:
            return ""
        if self.memory_policy is not None and not self.memory_policy(trigger.message.text):
            return ""
        try:
            return self.memory_provider(trigger) or ""
        except Exception:
            return ""

    def __call__(self, trigger: Trigger) -> str:
        turns = build_messages(
            trigger.context,
            trigger.message.text,
            profile=self.profile,
            fewshot=self.fewshot,
            limit=self.limit,
            memory_block=self._memory_block(trigger),
        )
        draft = clean_reply(self.draft_llm.chat(turns).text)
        if self.on_draft is not None:
            self.on_draft(draft)
        if not draft:
            return ""
        # 第二段失败就退回草稿，不能让整条链路崩
        try:
            rewritten = clean_reply(
                self.rewrite_llm.chat(
                    build_rewrite_messages(trigger.context, trigger.message.text, draft)
                ).text
            )
        except Exception:
            return draft
        return rewritten or draft
