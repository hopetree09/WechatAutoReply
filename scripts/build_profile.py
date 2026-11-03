"""从真实聊天记录自动生成「我的自述画像」（喂给运行时 prompt 的 data/profile.md）。

思路：
1. 从记忆库取「我」的真实发言，混合采样（近期的 + 高频口头禅 + 少量长回复）；
2. 让本地 7B 总结说话风格，输出固定格式的项目符号；
3. 写成 Markdown 文件（`load_profile` 直接读取）；LLM 失败时退回规则版画像。

用法：
    python scripts/build_profile.py --db data/agent.db --person 张伟 --out data/profile.md
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wagent.config import LlmSettings, get_settings  # noqa: E402
from wagent.llm.client import LLMClient, MessageTurn  # noqa: E402
from wagent.memory.store import MemoryStore  # noqa: E402
from wagent.memory.types import SELF_PERSON, PersonProfile  # noqa: E402

PROFILE_SYSTEM = """你是说话风格分析师。根据「我」的真实微信发言样本，总结出「我」的说话风格，
写成 Markdown 项目符号列表（每行一条，以 "- " 开头）。

必须覆盖这 8 类，且每条都要具体（带上真实例子/词）：
- 身份/职业线索：
- 口头禅（偶尔使用，不要每句都用，注明在什么场合用）：
- 常用表情/语气词：
- 说话习惯（长度、标点、是否用句号/波浪号）：
- 遇到夸我：
- 遇到求助：
- 话题偏好：
- 雷区/不爱聊：

隐私与质量要求（必须遵守）：
- 不要出现任何具体人名、医院/学校/单位、地址、账号、病史、金额、联系方式等隐私信息；
- 举例只写说话方式，不要引用敏感原文；
- 每条一行、尽量短；不确定的宁可不写。

只输出这 8 行，不要标题、不要解释、不要代码块。"""


def _stats(records) -> dict[str, float]:
    texts = [r.text.strip() for r in records if r.text.strip()]
    if not texts:
        return {"n": 0, "avg": 0.0, "median": 0.0, "period": 0.0, "emoji": 0.0}
    lengths = sorted(len(t) for t in texts)
    return {
        "n": len(texts),
        "avg": sum(lengths) / len(lengths),
        "median": float(lengths[len(lengths) // 2]),
        "period": sum(1 for t in texts if t.endswith("。")) / len(texts),
        "emoji": sum(1 for t in texts if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", t))
        / len(texts),
    }


def pick_samples(records, *, short: int = 120, long_max: int = 5, seed: int = 42) -> list[str]:
    """混合采样：近期短句 + 高频口头禅 + 少量长回复。"""
    rng = random.Random(seed)
    me = [r for r in records if r.speaker == "me" and 2 <= len(r.text.strip()) <= 60]
    rng.shuffle(me)
    picked = [r.text.strip() for r in me[:short]]

    freq = Counter(r.text.strip() for r in me if len(r.text.strip()) <= 8)
    for text, _count in freq.most_common(20):
        if text not in picked:
            picked.append(text)

    longs = [r.text.strip() for r in records if r.speaker == "me" and 60 < len(r.text.strip()) <= 200]
    rng.shuffle(longs)
    picked.extend(longs[:long_max])
    return picked


def fallback_profile(records, partner: str, facts: list[str]) -> str:
    """LLM 不可用时的规则版画像（也能用，只是不够细）。"""
    stats = _stats([r for r in records if r.speaker == "me"])
    freq = Counter(
        r.text.strip() for r in records if r.speaker == "me" and 1 <= len(r.text.strip()) <= 8
    )
    top = "、".join(text for text, _ in freq.most_common(8)) or "（无）"
    lines = [
        f"- 身份/职业线索：与「{partner}」是好友，聊天跨 {records[0].ts[:4]}~{records[-1].ts[:4]} 年。",
        f"- 口头禅（偶尔使用，不要每句都用，注明在什么场合用）：{top}",
        "- 常用表情/语气词：几乎没有 emoji，偶尔用 [强]/[衰] 这类微信表情。",
        f"- 说话习惯：平均 {stats['avg']:.1f} 字、中位数 {stats['median']:.0f} 字，"
        f"句号率仅 {stats['period'] * 100:.1f}%，基本不写标点，短句为主。",
        "- 遇到夸我：通常一两个字带过，不接话茬。",
        "- 遇到求助：直接给答案或反问细节，不说客套话。",
        "- 话题偏好：游戏、电脑硬件、学校/实验室、新闻热点。",
        "- 雷区/不爱聊：寒暄、长解释、说教。",
    ]
    if facts:
        lines.append("- 关于我的已知事实：" + "；".join(facts[:6]))
    return "\n".join(lines)


PRIVACY_PATTERNS = (
    r"医院", r"附院", r"[Cc][Tt]室", r"身份证", r"银行卡", r"密码", r"验证码",
    r"手机号", r"余额", r"转账", r"住院", r"病史", r"地址", r"单位",
)

REQUIRED_SECTIONS = ("身份", "口头禅", "表情", "说话习惯", "遇到夸", "遇到求助", "话题", "雷区")


def sanitize(text: str) -> str:
    """只保留 8 条以 "- " 开头的行，丢掉隐私/超长/非风格内容。"""
    lines = []
    for raw in text.replace("```markdown", "```").splitlines():
        line = raw.strip()
        if line.startswith("#") or line.startswith("```") or not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            body = line[2:].strip()
        elif re.match(r"^[一二三四五六七八九十\d]+[、.．]", line):
            body = re.sub(r"^[一二三四五六七八九十\d]+[、.．]\s*", "", line)
        else:
            continue
        if any(re.search(pattern, body) for pattern in PRIVACY_PATTERNS):
            continue  # 隐私内容不进画像
        if len(body) > 120:
            continue  # 过长的原文引用也不适合当画像
        lines.append("- " + body)
        if len(lines) >= 8:
            break
    return "\n".join(lines)


def has_all_sections(text: str) -> bool:
    return all(section in text for section in REQUIRED_SECTIONS)


def build_profile(
    *,
    db_path: Path,
    person: str,
    out_path: Path,
    samples: int = 120,
    model: str = "",
) -> str:
    store = MemoryStore(db_path)
    try:
        records = store.recent_messages(person, limit=200000)
        if not records:
            raise SystemExit(f"[!] 记忆库里没有 {person} 的聊天记录，先跑 import_chat")
        picks = pick_samples(records, short=samples)
        my_facts = [fact.value for fact in store.active_facts(SELF_PERSON, limit=10)]
        partner_facts = [fact.value for fact in store.active_facts(person, limit=10)]
    finally:
        store.close()

    stats = _stats([r for r in records if r.speaker == "me"])
    settings = get_settings()
    llm_settings = LlmSettings(
        provider="local",
        local_model=model or settings.llm.local_model,
    )
    prompt = [
        MessageTurn("system", PROFILE_SYSTEM),
        MessageTurn(
            "user",
            "统计："
            f"我的消息 {stats['n']:.0f} 条，平均 {stats['avg']:.1f} 字，中位数 {stats['median']:.0f} 字，"
            f"句号率 {stats['period'] * 100:.1f}%，emoji 率 {stats['emoji'] * 100:.1f}%。\n"
            f"聊天对象：{person}\n"
            f"关于我的事实：{'；'.join(my_facts) or '（暂无）'}\n"
            f"关于对方的事实：{'；'.join(partner_facts) or '（暂无）'}\n\n"
            "我的真实发言样本（每行一条）：\n" + "\n".join(picks),
        ),
    ]
    body = ""
    try:
        llm = LLMClient(llm_settings)
        result = llm.chat(prompt, temperature=0.3, max_tokens=800)
        body = sanitize(result.text)
    except Exception as exc:
        print(f"[profile] LLM 失败，退回规则版：{exc}")
    if len(body.splitlines()) < 5 or not has_all_sections(body):
        print('[profile] LLM 输出缺少必要小节或不可用，退回规则版画像')
        body = fallback_profile(records, person, my_facts)

    header = (
        "# 我的自述（自动生成）\n\n"
        f"> 由本地模型从 {stats['n']:.0f} 条真实发言中总结；重新生成：`python scripts/build_profile.py`\n\n"
    )
    content = header + body + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    # 同步一份结构化画像到记忆库，供检索/调试使用
    store = MemoryStore(db_path)
    try:
        store.upsert_profile(
            PersonProfile(
                person=SELF_PERSON,
                display_name="我",
                relation=f"与 {person} 聊天",
                traits=tuple(
                    line[2:] for line in body.splitlines() if line.startswith("- ")
                ),
                notes=body,
            )
        )
    finally:
        store.close()
    print(f"[profile] 写入 {out_path}（{len(body.splitlines())} 条）：")
    print(content)
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="从真实聊天记录生成我的自述画像")
    parser.add_argument("--db", default=str(ROOT / "data" / "agent.db"))
    parser.add_argument("--person", required=True, help="聊天对象（记忆库里的 person key）")
    parser.add_argument("--out", default=str(ROOT / "data" / "profile.md"))
    parser.add_argument("--samples", type=int, default=120, help="采样多少条我的发言")
    parser.add_argument("--model", default="", help="本地模型名（默认取配置）")
    args = parser.parse_args()
    build_profile(
        db_path=Path(args.db),
        person=args.person,
        out_path=Path(args.out),
        samples=args.samples,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
