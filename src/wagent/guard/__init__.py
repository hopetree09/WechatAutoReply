"""安全护栏：注入检测 / PII 脱敏 / 金钱话题拦截 / 输出校验。"""

from wagent.guard.policy import (
    GuardResult,
    check_output,
    has_injection,
    is_laughter,
    mask_pii,
)

__all__ = ["GuardResult", "check_output", "has_injection", "is_laughter", "mask_pii"]
