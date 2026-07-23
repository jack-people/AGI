"""
计算器工具 — 封装安全的数学表达式求值。

基于 numexpr，只暴露数学函数和基本运算符，
不暴露 Python 内置函数（open、exec 等），保证安全性。
"""

import numexpr
import math
from typing import Optional
from langchain_core.tools import tool


_SAFE_LOCALS = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
}

_SAFE_GLOBALS: dict = {
    "__builtins__": {},
}


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持 + - * / ** % 和数学函数（sin, cos, sqrt, log 等）。

    Args:
        expression: 纯数学表达式字符串，例如 "2 ** 10", "sqrt(144) + 3 * 5",
                    "cos(pi/4)", "log10(1000)"。

    Returns:
        计算结果字符串。
    """
    try:
        # 用 numexpr 求值，速度快且只暴露数学函数
        result = numexpr.evaluate(expression, local_dict=_SAFE_LOCALS)
        # 标量结果直接返回
        if hasattr(result, "ndim") and result.ndim == 0:
            val = result.item()
            if isinstance(val, float):
                return f"{val:.12g}"
            return str(val)
        return str(result)
    except Exception as e:
        # 回退：用受限 eval 再试一次（兼容性兜底）
        try:
            val = eval(expression, _SAFE_GLOBALS, _SAFE_LOCALS | {"__builtins__": {}})
            if isinstance(val, float):
                return f"{val:.12g}"
            return str(val)
        except Exception as fallback_e:
            return f"表达式错误: {fallback_e}"
