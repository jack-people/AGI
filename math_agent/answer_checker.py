"""
答案判定 — 从 Agent 的自然语言输出中提取数值并对比预期。
"""

import re
from typing import Any, Optional, Tuple, Union


def _extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数字（含小数和负号）。"""
    tokens: list[float] = []
    for m in re.finditer(r"-?\d+\.?\d*", text):
        try:
            tokens.append(float(m.group()))
        except ValueError:
            pass
    return tokens


def _extract_percentages(text: str) -> list[float]:
    """提取百分数，去掉 % 号转为纯数字。"""
    values: list[float] = []
    for m in re.finditer(r"(\d+\.?\d*)\s*%", text):
        try:
            values.append(float(m.group(1)))
        except ValueError:
            pass
    return values


def _check_accept(text: str, accept_list: list) -> bool:
    """检查文本是否匹配可接受答案列表中的任意一项。"""
    text_lower = text.lower()
    for item in accept_list:
        if isinstance(item, str):
            if item.lower() in text_lower:
                return True
        elif isinstance(item, (int, float)):
            if item in _extract_numbers(text):
                return True
    return False


def check_answer(
    agent_output: str,
    answer: dict,
    tolerance: float = 1e-4,
) -> Tuple[bool, str]:
    """判定 Agent 输出是否与预期答案一致。

    Args:
        agent_output: Agent 的 Final Answer 文本。
        answer: 问题中的 answer 字段（含 exact/approx/accept）。
        tolerance: 浮点容差。

    Returns:
        (是否通过, 判据说明)
    """
    numbers = _extract_numbers(agent_output)
    percentages = _extract_percentages(agent_output)

    # 1) 检查 accept 列表
    accept_list = answer.get("accept", [])
    if accept_list:
        if _check_accept(agent_output, accept_list):
            return True, f"匹配 accept 列表: {accept_list}"

    # 2) 近似匹配
    if "approx" in answer:
        expected = float(answer["approx"])
        for val in numbers:
            if abs(val - expected) <= tolerance:
                return True, f"近似匹配: {expected} (容差 {tolerance})"
        # 百分数匹配（把百分数转成小数）
        for val in percentages:
            if abs(val - expected) <= tolerance:
                return True, f"近似匹配 (百分数): {val}% == {expected}"
        return False, f"未匹配近似值 {expected}，提取到的数值: {numbers}"

    # 3) 精确匹配
    if "exact" in answer:
        expected = float(answer["exact"])
        for val in numbers:
            if abs(val - expected) <= tolerance:
                return True, f"精确匹配: {expected}"
        # 百分数检查
        for val in percentages:
            if abs(val - expected) <= tolerance:
                return True, f"精确匹配 (百分数): {val}%"
        return False, f"未匹配精确值 {expected}，提取到的数值: {numbers}"

    return False, "答案格式无法识别"


if __name__ == "__main__":
    # 简单冒烟测试
    tests = [
        ("答案是 42", {"exact": 42}, True),
        ("面积约为 78.54 平方单位", {"approx": 78.54}, True),
        ("3/4 等于 75%", {"exact": 75}, True),
        ("x = 5 和 y = 3", {"exact": 5}, True),
    ]
    for output, answer, expected in tests:
        ok, reason = check_answer(output, answer)
        status = "PASS" if ok == expected else "FAIL"
        print(f"[{status}] {reason}")
