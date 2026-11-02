"""合成聊天数据：给 LoRA 训练管线做端到端验证（不碰真实聊天记录）。

真实数据到位前，用它验证「数据格式 → 训练 → 推理 → A/B 评测」整条链路能跑通；
数据量、风格模板都可通过参数控制。真实数据集只要换成
``python -m wagent.import_chat`` 导入的记忆库即可，格式完全一致。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from wagent.memory.types import MessageRecord

# (对方的话, 我的回复) —— 模拟"话不多、爱用口头禅、偶尔冷幽默"的人设
EXCHANGES: tuple[tuple[str, str], ...] = (
    ("在吗", "在的"),
    ("在干嘛", "摸鱼呢"),
    ("今晚有空吗", "有啊 咋了"),
    ("周末一起打球？", "行 几点"),
    ("明天几点到", "九点吧 我早点过去"),
    ("吃饭了吗", "刚吃完 你呢"),
    ("最近怎么样", "还行 忙成狗"),
    ("我换工作了", "哦？去哪了"),
    ("我在字节实习", "可以啊 大厂"),
    ("今天加班到十点", "惨 早点睡"),
    ("帮我看看这个简历", "发我 我看下"),
    ("这个报错咋解决", "你先把日志贴出来"),
    ("你上次说的那个方案呢", "在写了 明天给你"),
    ("周末有空帮我搬家吗", "周六可以 周日不行"),
    ("我养了两只猫", "橘猫吗 肯定很胖"),
    ("我下个月搬去滨江", "行 到时候喊我"),
    ("最近在学什么", "在看 agent 那套东西"),
    ("推荐个电影", "最近没看 你先说"),
    ("这个项目靠谱吗", "看需求 不急着上"),
    ("帮我改个文案", "你把原文发我"),
    ("你怎么看这个事", "我倾向于先做小范围验证"),
    ("在忙吗", "还好 说"),
    ("晚上一起吃饭？", "可以 你定地方"),
    ("帮我带杯咖啡", "美式？"),
    ("你几点下班", "六点半 差不多"),
    ("方案我改完了", "好 我看看"),
    ("这个价格合理吗", "偏高 再压一压"),
    ("我有点焦虑", "别想太多 先睡一觉"),
    ("今天面试挂了", "正常 下一个更好"),
    ("我买了新电脑", "啥配置"),
    ("北京下雨了", "带伞了吗"),
    ("你周末干嘛", "宅着 补觉"),
    ("这个库怎么用", "文档里有例子 我发你"),
    ("你能帮我写个脚本吗", "可以 需求说清楚"),
    ("我辞职了", "想好了就行"),
    ("出来喝一杯？", "行 老地方"),
    ("我脱单了", "可以啊 恭喜"),
    ("你觉得我行不行", "行 别怂"),
    ("帮我看下这段代码", "发过来"),
    ("明天开会吗", "开 十点"),
    ("这个你看过没", "没 发我看看"),
    ("今天好累", "早点休息"),
    ("你睡了吗", "还没"),
    ("帮忙取个快递", "放哪了"),
    ("我改天请你吃饭", "行 记着了"),
    ("这个周末有雨", "那就在家待着"),
    ("你还会做饭呢", "会一点 不好吃"),
)

FILLERS: tuple[str, ...] = (
    "嗯",
    "好",
    "行",
    "哈哈",
    "确实",
    "然后呢",
    "可以",
    "我看看",
    "收到",
)

EMOJIS: tuple[str, ...] = ("😂", "🐶", "👍", "🤣", "🙂", "😅")


@dataclass(frozen=True, slots=True)
class SynthConfig:
    """合成数据参数。"""

    exchanges: int = 320
    seed: int = 7
    context_turns: int = 3
    emoji_rate: float = 0.25
    filler_rate: float = 0.2
    person: str = "张伟"


def _add_emoji(reply: str, rng: random.Random, rate: float) -> str:
    if rng.random() < rate:
        return reply + rng.choice(EMOJIS)
    return reply


def generate_records(config: SynthConfig | None = None) -> list[MessageRecord]:
    """生成按时间递增的合成聊天记录（含上下文轮次与偶发语气词）。"""
    config = config or SynthConfig()
    rng = random.Random(config.seed)
    records: list[MessageRecord] = []
    # 打乱顺序、循环使用模板，保证组合多样
    pool = list(EXCHANGES)
    rng.shuffle(pool)

    base_minutes = 0
    for exchange_index in range(config.exchanges):
        # 先塞几轮闲聊当上下文
        for _turn in range(rng.randint(0, config.context_turns)):
            who = "me" if rng.random() < 0.4 else "other"
            text = rng.choice(FILLERS) if rng.random() < config.filler_rate else rng.choice(
                [pair[0] for pair in EXCHANGES]
            )
            base_minutes += rng.randint(1, 20)
            records.append(
                _record(config.person, who, text, base_minutes)
            )

        other_text, my_reply = pool[exchange_index % len(pool)]
        base_minutes += rng.randint(1, 10)
        records.append(_record(config.person, "other", other_text, base_minutes))
        base_minutes += rng.randint(0, 2)
        records.append(
            _record(
                config.person,
                "me",
                _add_emoji(my_reply, rng, config.emoji_rate),
                base_minutes,
            )
        )
    return records


def _record(person: str, speaker: str, text: str, minute_offset: int) -> MessageRecord:
    hours, minutes = divmod(minute_offset, 60)
    days, hours = divmod(hours, 24)
    return MessageRecord(
        person=person,
        speaker=speaker,
        text=text,
        ts=f"2026-10-{1 + days:02d}T{hours:02d}:{minutes:02d}:00+08:00",
        session=person,
        source="synth",
    )
