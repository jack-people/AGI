"""
100 道数学题 — 按类别组织，每道题带预期答案。

Answer 字段：
  - exact: 精确值（整数或浮点数）
  - approx: 近似值容差（仅用于浮点对比）
  - accept: 额外可接受的答案（列表）
  - units: 单位说明
"""

import math
from typing import Any, Dict, List, Optional


Problem = Dict[str, Any]


def _problems() -> List[Problem]:
    return [
        # ═══════════════════════════════════════
        # 1. 基础算术 (1-25)
        # ═══════════════════════════════════════
        {"id": "A01", "category": "arithmetic", "question": "12 + 37 等于多少？", "answer": {"exact": 49}},
        {"id": "A02", "category": "arithmetic", "question": "85 - 29 等于多少？", "answer": {"exact": 56}},
        {"id": "A03", "category": "arithmetic", "question": "14 × 6 等于多少？", "answer": {"exact": 84}},
        {"id": "A04", "category": "arithmetic", "question": "156 ÷ 12 等于多少？", "answer": {"exact": 13}},
        {"id": "A05", "category": "arithmetic", "question": "2 的 10 次方是多少？", "answer": {"exact": 1024}},
        {"id": "A06", "category": "arithmetic", "question": "144 的平方根是多少？", "answer": {"exact": 12}},
        {"id": "A07", "category": "arithmetic", "question": "3 的 5 次方是多少？", "answer": {"exact": 243}},
        {"id": "A08", "category": "arithmetic", "question": "3 + 4 × 2 等于多少？", "answer": {"exact": 11}},
        {"id": "A09", "category": "arithmetic", "question": "(3 + 4) × 2 等于多少？", "answer": {"exact": 14}},
        {"id": "A10", "category": "arithmetic", "question": "50 - 3 × 8 + 4 等于多少？", "answer": {"exact": 30}},
        {"id": "A11", "category": "arithmetic", "question": "100 ÷ 4 × 5 等于多少？", "answer": {"exact": 125}},
        {"id": "A12", "category": "arithmetic", "question": "2 × 3 + 4 × 5 等于多少？", "answer": {"exact": 26}},
        {"id": "A13", "category": "arithmetic", "question": "7!（7 的阶乘）是多少？", "answer": {"exact": 5040}},
        {"id": "A14", "category": "arithmetic", "question": "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 等于多少？", "answer": {"exact": 55}},
        {"id": "A15", "category": "arithmetic", "question": "99 + 98 + 97 等于多少？", "answer": {"exact": 294}},
        {"id": "A16", "category": "arithmetic", "question": "15 × 15 等于多少？", "answer": {"exact": 225}},
        {"id": "A17", "category": "arithmetic", "question": "999 + 1 等于多少？", "answer": {"exact": 1000}},
        {"id": "A18", "category": "arithmetic", "question": "1000 - 456 等于多少？", "answer": {"exact": 544}},
        {"id": "A19", "category": "arithmetic", "question": "24 × 25 等于多少？", "answer": {"exact": 600}},
        {"id": "A20", "category": "arithmetic", "question": "125 × 8 等于多少？", "answer": {"exact": 1000}},
        {"id": "A21", "category": "arithmetic", "question": "5 的 4 次方是多少？", "answer": {"exact": 625}},
        {"id": "A22", "category": "arithmetic", "question": "2 + 2 × 2 + 2 等于多少？", "answer": {"exact": 8}},
        {"id": "A23", "category": "arithmetic", "question": "81 的平方根是多少？", "answer": {"exact": 9}},
        {"id": "A24", "category": "arithmetic", "question": "3 的 0 次方是多少？", "answer": {"exact": 1}},
        {"id": "A25", "category": "arithmetic", "question": "13 × 17 等于多少？", "answer": {"exact": 221}},
        # ═══════════════════════════════════════
        # 2. 代数 (26-45)
        # ═══════════════════════════════════════
        {"id": "B01", "category": "algebra", "question": "如果 3x + 7 = 22，x 等于多少？", "answer": {"exact": 5}},
        {"id": "B02", "category": "algebra", "question": "如果 2x - 5 = 13，x 等于多少？", "answer": {"exact": 9}},
        {"id": "B03", "category": "algebra", "question": "如果 5x + 3 = 3x + 11，x 等于多少？", "answer": {"exact": 4}},
        {"id": "B04", "category": "algebra", "question": "如果 4(x + 2) = 24，x 等于多少？", "answer": {"exact": 4}},
        {"id": "B05", "category": "algebra", "question": "如果 x² = 49，x 的正数解是多少？", "answer": {"exact": 7}},
        {"id": "B06", "category": "algebra", "question": "如果 2x + 1 = 17，x 等于多少？", "answer": {"exact": 8}},
        {"id": "B07", "category": "algebra", "question": "如果 6x - 8 = 4x + 6，x 等于多少？", "answer": {"exact": 7}},
        {"id": "B08", "category": "algebra", "question": "如果 7x = 84，x 等于多少？", "answer": {"exact": 12}},
        {"id": "B09", "category": "algebra", "question": "如果 x/5 + 3 = 8，x 等于多少？", "answer": {"exact": 25}},
        {"id": "B10", "category": "algebra", "question": "如果 3(x - 4) = 2x + 1，x 等于多少？", "answer": {"exact": 13}},
        {"id": "B11", "category": "algebra", "question": "如果 x + y = 10 且 x - y = 4，那么 x 和 y 分别是多少？", "answer": {"exact": 7, "accept": [[7, 3], "x=7,y=3"]}},
        {"id": "B12", "category": "algebra", "question": "如果 2x + y = 16 且 x = 5，y 等于多少？", "answer": {"exact": 6}},
        {"id": "B13", "category": "algebra", "question": "如果 5x - 20 = 0，x 等于多少？", "answer": {"exact": 4}},
        {"id": "B14", "category": "algebra", "question": "如果 2x + 3 = 15，x 等于多少？", "answer": {"exact": 6}},
        {"id": "B15", "category": "algebra", "question": "如果 x + 2x + 3x = 48，x 等于多少？", "answer": {"exact": 8}},
        {"id": "B16", "category": "algebra", "question": "如果 8x - 4 = 36，x 等于多少？", "answer": {"exact": 5}},
        {"id": "B17", "category": "algebra", "question": "如果 10x + 15 = 5x + 45，x 等于多少？", "answer": {"exact": 6}},
        {"id": "B18", "category": "algebra", "question": "如果 x/3 = 7，x 等于多少？", "answer": {"exact": 21}},
        {"id": "B19", "category": "algebra", "question": "如果 4x - 7 = 2x + 9，x 等于多少？", "answer": {"exact": 8}},
        {"id": "B20", "category": "algebra", "question": "如果 x + 5 = 12，x 等于多少？", "answer": {"exact": 7}},
        # ═══════════════════════════════════════
        # 3. 几何 (46-65)
        # ═══════════════════════════════════════
        {"id": "C01", "category": "geometry", "question": "边长为 5 的正方形面积是多少？", "answer": {"exact": 25}},
        {"id": "C02", "category": "geometry", "question": "边长为 5 的正方形周长是多少？", "answer": {"exact": 20}},
        {"id": "C03", "category": "geometry", "question": "长 12 宽 8 的长方形面积是多少？", "answer": {"exact": 96}},
        {"id": "C04", "category": "geometry", "question": "长 12 宽 8 的长方形周长是多少？", "answer": {"exact": 40}},
        {"id": "C05", "category": "geometry", "question": "半径为 5 的圆的面积大约是多少？（保留两位小数）", "answer": {"approx": 78.54}},
        {"id": "C06", "category": "geometry", "question": "半径为 5 的圆的周长大约是多少？（保留两位小数）", "answer": {"approx": 31.42}},
        # (C05 和 C06 实际用 π 算，approx 已保留两位小数)
        {"id": "C07", "category": "geometry", "question": "底为 6、高为 4 的三角形面积是多少？", "answer": {"exact": 12}},
        {"id": "C08", "category": "geometry", "question": "半径为 10 的圆的面积大约是多少？", "answer": {"approx": 314.16}},
        {"id": "C09", "category": "geometry", "question": "边长为 3 的立方体体积是多少？", "answer": {"exact": 27}},
        {"id": "C10", "category": "geometry", "question": "半径为 3、高为 5 的圆柱体体积大约是多少？（保留整数）", "answer": {"approx": 141.37}},
        {"id": "C11", "category": "geometry", "question": "上底 4、下底 6、高 3 的梯形面积是多少？", "answer": {"exact": 15}},
        {"id": "C12", "category": "geometry", "question": "边长为 10 的正方体表面积是多少？", "answer": {"exact": 600}},
        {"id": "C13", "category": "geometry", "question": "直径为 8 的圆的周长大约是多少？（保留两位小数）", "answer": {"approx": 25.13}},
        {"id": "C14", "category": "geometry", "question": "底为 5、高为 8 的平行四边形面积是多少？", "answer": {"exact": 40}},
        {"id": "C15", "category": "geometry", "question": "半径为 4 的球体体积大约是多少？（保留整数）", "answer": {"approx": 268.08}},
        {"id": "C16", "category": "geometry", "question": "边长为 6 的正六边形面积大约是多少？（保留整数）", "answer": {"approx": 93.53}},
        {"id": "C17", "category": "geometry", "question": "一个长方形的长是宽的 2 倍，若宽为 4，面积是多少？", "answer": {"exact": 32}},
        {"id": "C18", "category": "geometry", "question": "一个正方形边长增加 2 倍后，新面积是原来的多少倍？", "answer": {"exact": 9}},
        {"id": "C19", "category": "geometry", "question": "半径 7 的圆的面积大约是多少？（保留一位小数）", "answer": {"approx": 153.9}},
        {"id": "C20", "category": "geometry", "question": "一个边长为 4 的正方形对角线长度大约是多少？（保留两位小数）", "answer": {"approx": 5.66}},
        # ═══════════════════════════════════════
        # 4. 百分数 / 分数 (66-80)
        # ═══════════════════════════════════════
        {"id": "D01", "category": "percent", "question": "200 的 15% 是多少？", "answer": {"exact": 30}},
        {"id": "D02", "category": "percent", "question": "120 的 25% 是多少？", "answer": {"exact": 30}},
        {"id": "D03", "category": "percent", "question": "一件商品原价 80 元，打 7 折后多少钱？", "answer": {"exact": 56}},
        {"id": "D04", "category": "percent", "question": "500 的 12% 是多少？", "answer": {"exact": 60}},
        {"id": "D05", "category": "percent", "question": "30 是 150 的百分之多少？", "answer": {"exact": 20, "accept": ["20%"]}},
        {"id": "D06", "category": "percent", "question": "一件商品提价 20% 后是 120 元，原价是多少？", "answer": {"exact": 100}},
        {"id": "D07", "category": "percent", "question": "1/4 化成百分数是多少？", "answer": {"exact": 25, "accept": ["25%"]}},
        {"id": "D08", "category": "percent", "question": "3/5 化成百分数是多少？", "answer": {"exact": 60, "accept": ["60%"]}},
        {"id": "D09", "category": "percent", "question": "50 增加 10% 后是多少？", "answer": {"exact": 55}},
        {"id": "D10", "category": "percent", "question": "80 减少 25% 后是多少？", "answer": {"exact": 60}},
        {"id": "D11", "category": "percent", "question": "1/3 + 1/6 等于多少？（用分数表示）", "answer": {"exact": 0.5, "accept": ["1/2", "0.5"]}},
        {"id": "D12", "category": "percent", "question": "2/3 约等于多少？（保留三位小数）", "answer": {"approx": 0.667}},
        {"id": "D13", "category": "percent", "question": "5/8 化成小数是多少？", "answer": {"exact": 0.625}},
        {"id": "D14", "category": "percent", "question": "3/4 + 1/8 等于多少？", "answer": {"exact": 0.875, "accept": ["7/8", "0.875"]}},
        {"id": "D15", "category": "percent", "question": "一个班级 40 人，女生占 55%，女生有多少人？", "answer": {"exact": 22}},
        # ═══════════════════════════════════════
        # 5. 应用题 (81-100)
        # ═══════════════════════════════════════
        {"id": "E01", "category": "word", "question": "小明有 15 个苹果，给了小红 7 个，又买了 5 个，现在有几个？", "answer": {"exact": 13}},
        {"id": "E02", "category": "word", "question": "一本书 240 页，小华每天读 15 页，需要几天读完？", "answer": {"exact": 16}},
        {"id": "E03", "category": "word", "question": "甲乙两地相距 360 公里，一辆车以每小时 80 公里的速度行驶，需要几小时？", "answer": {"exact": 4.5}},
        {"id": "E04", "category": "word", "question": "一个水池有 3 个进水管，每个每小时进水 4 立方米，2 小时共进水多少立方米？", "answer": {"exact": 24}},
        {"id": "E05", "category": "word", "question": "一斤苹果 5 元，买 3.5 斤需要多少钱？", "answer": {"exact": 17.5}},
        {"id": "E06", "category": "word", "question": "一个长方形的花圃长 15 米，宽 10 米，每平方米种 4 株花，一共种多少株？", "answer": {"exact": 600}},
        {"id": "E07", "category": "word", "question": "甲有 100 元，乙比甲多 20%，乙有多少元？", "answer": {"exact": 120}},
        {"id": "E08", "category": "word", "question": "一项工程 8 天完成，每天完成这项工程的几分之几？", "answer": {"exact": 0.125, "accept": ["1/8", "12.5%"]}},
        {"id": "E09", "category": "word", "question": "一个三角形三边长分别是 3、4、5，它的周长是多少？", "answer": {"exact": 12}},
        {"id": "E10", "category": "word", "question": "一本书打 8 折后是 64 元，原价是多少元？", "answer": {"exact": 80}},
        {"id": "E11", "category": "word", "question": "小李每分钟走 70 米，走 1050 米需要几分钟？", "answer": {"exact": 15}},
        {"id": "E12", "category": "word", "question": "一个教室长 9 米宽 6 米高 3 米，它的体积是多少立方米？", "answer": {"exact": 162}},
        {"id": "E13", "category": "word", "question": "甲乙两人共有 60 元，甲是乙的 2 倍，甲有多少元？", "answer": {"exact": 40}},
        {"id": "E14", "category": "word", "question": "一箱牛奶 12 盒，每盒 250 毫升，一共多少毫升？", "answer": {"exact": 3000}},
        {"id": "E15", "category": "word", "question": "一个工人每小时做 8 个零件，一天工作 7.5 小时，能做多少个？", "answer": {"exact": 60}},
        {"id": "E16", "category": "word", "question": "小张的月薪是 5000 元，每月储蓄 20%，一年储蓄多少元？", "answer": {"exact": 12000}},
        {"id": "E17", "category": "word", "question": "一块正方形菜地边长 12 米，四周围上篱笆需要多少米？", "answer": {"exact": 48}},
        {"id": "E18", "category": "word", "question": "某班有 48 人，今天缺席 3 人，出勤率是多少？（百分比保留整数）", "answer": {"approx": 93.75, "accept": ["93.75%", "94%"]}},
        {"id": "E19", "category": "word", "question": "一个圆形花坛半径 3 米，绕花坛走一圈大约多少米？（保留两位小数）", "answer": {"approx": 18.85}},
        {"id": "E20", "category": "word", "question": "小明考了 85 分，比上次提高了 15 分，提高了百分之多少？（保留一位小数）", "answer": {"approx": 21.4, "accept": ["21.4%", "21.43%"]}},
    ]


def get_problems() -> List[Problem]:
    return _problems()


def get_problems_by_category(category: str) -> List[Problem]:
    return [p for p in _problems() if p["category"] == category]


def count_problems() -> int:
    return len(_problems())


if __name__ == "__main__":
    all_problems = get_problems()
    print(f"共 {len(all_problems)} 道题")
    for cat in sorted({p["category"] for p in all_problems}):
        sub = get_problems_by_category(cat)
        names = {"arithmetic": "基础算术", "algebra": "代数", "geometry": "几何", "percent": "百分数/分数", "word": "应用题"}
        print(f"  {names.get(cat, cat)}: {len(sub)} 题")
