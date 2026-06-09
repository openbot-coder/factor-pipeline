"""Shared utility functions for factor_pipeline."""

from __future__ import annotations


def norm_code(code: str) -> str | None:
    """规范化 A 股代码：'000001' → '000001.SZSE'

    Returns:
        标准化的 symbol 字符串，无法识别时返回 None。
    """
    c = str(code).strip().zfill(6)
    if c.startswith(("60", "68", "90")):
        return f"{c}.SSE"
    elif c.startswith(("00", "30", "20")):
        return f"{c}.SZSE"
    elif c.startswith(("43", "83", "87", "92")):
        return f"{c}.BSE"
    return None
