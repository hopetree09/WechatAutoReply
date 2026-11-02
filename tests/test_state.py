"""触发状态机单测。"""

from __future__ import annotations

from wagent.agent.state import ConversationState, Phase
from wagent.perception.types import Message, Speaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def msg(speaker: Speaker, text: str) -> Message:
    from wagent.perception.types import Box

    return Message(speaker=speaker, text=text, box=Box(0, 0, 10, 10))


def other(text: str) -> Message:
    return msg(Speaker.OTHER, text)


def me(text: str) -> Message:
    return msg(Speaker.ME, text)


def make_state(clock: FakeClock | None = None, **kwargs) -> ConversationState:
    return ConversationState(_clock=clock or FakeClock(), **kwargs)


# ---------------------------------------------------------------- 基线


def test_first_frame_establishes_baseline_without_reply() -> None:
    state = make_state()
    assert state.observe([me("早"), other("在吗")], session="张三") is None
    assert state.phase is Phase.IDLE
    assert state.baseline_done
    assert state.session == "张三"


def test_new_other_message_triggers() -> None:
    state = make_state()
    state.observe([other("在吗")], session="张三")
    trigger = state.observe([other("在吗"), other("今晚有空吗")], session="张三")
    assert trigger is not None
    assert trigger.message.text == "今晚有空吗"
    assert trigger.session == "张三"


def test_same_frame_does_not_trigger_again() -> None:
    state = make_state()
    state.observe([other("在吗")], session="张三")
    assert state.observe([other("在吗"), other("你好")], session="张三") is not None
    state.mark_sent("在的")
    # 回复还没出现在屏幕上，重复喂同一帧
    assert state.observe([other("在吗"), other("你好")], session="张三") is None


# ---------------------------------------------------------------- 防抖


def test_middle_flicker_does_not_trigger() -> None:
    """滚动/OCR 抖动导致中间某条消息变化，但底部没变，不应该再触发。"""
    state = make_state()
    state.observe([other("A"), other("B")], session="s")
    assert state.observe([other("A"), other("B"), other("C")], session="s") is not None
    state.mark_sent("好的")
    state.observe([other("A"), other("B"), other("C"), me("好的")], session="s")  # 回复出现
    # 中间消息变了、底部没变
    assert state.observe([other("A2"), other("B"), other("C"), me("好的")], session="s") is None


def test_bottom_is_me_no_trigger() -> None:
    state = make_state()
    state.observe([other("在吗")], session="s")
    assert state.observe([other("在吗"), me("在的")], session="s") is None


def test_scrolling_does_not_trigger() -> None:
    """滚动导致旧消息上移/掉出窗口、但末尾没有新消息 → 不触发。"""
    state = make_state()
    state.observe([other("1"), other("2"), other("3"), other("4")], session="s")
    assert state.observe([other("2"), other("3"), other("4")], session="s") is None
    assert state.observe([other("3"), other("4")], session="s") is None


# ---------------------------------------------------------------- 回复 / 冷却


def test_reply_echo_triggers_cooldown_then_allows_next() -> None:
    clock = FakeClock()
    state = make_state(clock, cooldown_s=8.0)
    state.observe([other("在吗")], session="s")
    trigger = state.observe([other("在吗"), other("几点")], session="s")
    assert trigger is not None
    assert state.phase is Phase.REPLYING

    state.mark_sent("八点吧")
    assert state.observe([other("在吗"), other("几点"), me("八点吧")], session="s") is None
    assert state.phase is Phase.COOLDOWN

    # 冷却期内对方又发一条 → 不触发
    assert state.observe([other("几点"), me("八点吧"), other("好")], session="s") is None
    clock.advance(9.0)
    # 冷却结束后再次变化才触发
    assert state.observe([me("八点吧"), other("好"), other("那就这样")], session="s") is not None


def test_repeated_identical_message_triggers_again() -> None:
    """对方第二次发一模一样的「哈哈哈」，仍然要回（旧实现会被永久去重吞掉）。"""
    clock = FakeClock()
    state = make_state(clock, cooldown_s=0.0)
    state.observe([other("哈哈哈")], session="s")
    assert state.observe([other("哈哈哈"), other("哈哈哈")], session="s") is not None
    state.mark_sent("笑死")
    state.observe([other("哈哈哈"), other("哈哈哈"), me("笑死")], session="s")
    clock.advance(1.0)
    trigger = state.observe([other("哈哈哈"), me("笑死"), other("哈哈哈")], session="s")
    assert trigger is not None
    assert trigger.message.text == "哈哈哈"


def test_reply_timeout_returns_to_idle_and_can_retry() -> None:
    clock = FakeClock()
    state = make_state(clock, reply_timeout_s=20.0)
    state.observe([other("在吗")], session="s")
    assert state.observe([other("在吗"), other("喂")], session="s") is not None
    # 回复一直没有出现在屏幕上
    clock.advance(21.0)
    assert state.observe([other("在吗"), other("喂")], session="s") is None
    assert state.phase is Phase.IDLE
    # fail() 清空尾部后，下一帧可以重新评估同一条消息
    trigger = state.observe([other("在吗"), other("喂")], session="s")
    assert trigger is not None


def test_skip_allows_next_message() -> None:
    state = make_state()
    state.observe([other("嗯")], session="s")
    assert state.observe([other("嗯"), other("表情包")], session="s") is not None
    state.skip()
    assert state.phase is Phase.IDLE
    assert state.observe([other("嗯"), other("表情包"), other("在吗")], session="s") is not None


# ---------------------------------------------------------------- 会话切换


def test_session_switch_resets_baseline() -> None:
    state = make_state()
    state.observe([other("张三的消息")], session="张三")
    # 切到李四的聊天：历史消息不能触发回复
    assert state.observe([other("李四的历史消息")], session="李四") is None
    assert state.session == "李四"
    assert state.baseline_done
    # 新会话里来了新消息才会触发
    assert state.observe([other("李四的历史消息"), other("新消息")], session="李四") is not None


def test_empty_session_does_not_reset() -> None:
    state = make_state()
    state.observe([other("在吗")], session="张三")
    # OCR 偶尔读不到会话名，此时不应该被当成切会话
    assert state.observe([other("在吗"), other("喂")], session="") is not None


# ---------------------------------------------------------------- 上下文


def test_multiple_messages_between_polls_trigger_once() -> None:
    """轮询间隔内对方连发多条：只触发一次，上下文包含全部新消息。"""
    state = make_state(context_len=6)
    state.observe([other("在吗")], session="s")
    trigger = state.observe([other("在吗"), other("今晚"), other("有空吗")], session="s")
    assert trigger is not None
    assert trigger.message.text == "有空吗"
    assert [m.text for m in trigger.context][-3:] == ["在吗", "今晚", "有空吗"]


def test_context_is_limited() -> None:
    state = make_state(context_len=3)
    state.observe([other("1")], session="s")
    trigger = state.observe([other("1"), other("2"), other("3"), other("4")], session="s")
    assert trigger is not None
    assert len(trigger.context) == 3
    assert trigger.message.text == "4"


def test_snapshot_contains_phase() -> None:
    state = make_state()
    state.observe([other("在吗")], session="s")
    snap = state.snapshot()
    assert snap["phase"] == Phase.IDLE.value
    assert snap["baseline_done"] is True
