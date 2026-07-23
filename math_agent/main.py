"""
主入口 — 交互式数学 Agent。

用法：
  python main.py                                    # 从 .env 读取 API Key
  python main.py --api-key sk-xxx                    # 直接传入 API Key
  python main.py --api-base http://localhost:11434/v1 --model qwen2.5:7b
"""

import argparse
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

from agent import build_agent


# ----- 自定义回调：捕获中间推理过程 -----

class ReasoningCallback(BaseCallbackHandler):
    """捕获 Agent 的 Thought/Action/Observation 并友好打印。"""

    def __init__(self) -> None:
        super().__init__()
        self.steps: List[str] = []

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        **kwargs: Any,
    ) -> None:
        tool = action.tool
        tool_input = action.tool_input
        log = action.log or ""

        # 从 log 里提取 Thought，若无则从 action 合成
        thought = ""
        for line in log.strip().split("\n"):
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
                break
        if not thought and log.strip():
            thought = log.strip()[:200].rstrip()

        step_lines = []
        if thought:
            step_lines.append(f"  💭 Thought: {thought}")
        step_lines.append(f"  🔧 Action: {tool}({tool_input})")

        entry = "\n".join(step_lines)
        self.steps.append(entry)
        print(f"\n{entry}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        **kwargs: Any,
    ) -> None:
        print(f"  📊 Observation: {output}")


# ----- CLI -----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangChain 数学 Agent — 带计算器工具，展示中间推理过程",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API Key（缺省从 OPENAI_API_KEY 环境变量读取）",
    )
    parser.add_argument(
        "--api-base",
        help="自定义 API 地址，例如 http://localhost:11434/v1（Ollama）",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="模型名称（默认 gpt-4o-mini）",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="直接传入问题，不进入交互模式",
    )
    return parser.parse_args()


PRESET_QUESTIONS = [
    "1 + 1 等于几？",
    "3 * 7 + 12 / 4 等于多少？",
    "半径为 5 的圆的面积是多少？",
    "如果 3x + 7 = 22，x 等于多少？",
    "一个长方形的长是 12 厘米，宽是 8 厘米，它的周长和面积各是多少？",
]


def print_presets() -> None:
    print("\n预设问题：")
    for i, q in enumerate(PRESET_QUESTIONS, 1):
        print(f"  [{i}] {q}")
    print("  [0] 输入自定义问题")


def interactive_loop(agent: Any, callback: ReasoningCallback) -> None:
    print("\n" + "=" * 60)
    print("🧮  Math Agent — 输入 q 退出")
    print("=" * 60)

    while True:
        print_presets()
        choice = input("\n请选择 (0-5, 或 q 退出): ").strip()

        if choice.lower() == "q":
            print("再见！")
            break

        try:
            idx = int(choice)
        except ValueError:
            print("无效输入，请输入数字。")
            continue

        if idx == 0:
            question = input("请输入你的数学问题: ").strip()
        elif 1 <= idx <= len(PRESET_QUESTIONS):
            question = PRESET_QUESTIONS[idx - 1]
        else:
            print("无效选项。")
            continue

        if not question:
            print("问题不能为空。")
            continue

        print(f"\n{'─' * 60}")
        print(f"❓ Question: {question}")
        print(f"{'─' * 60}")

        callback.steps.clear()
        try:
            result = agent.invoke(
                {"input": question},
                {"callbacks": [callback]},
            )
            print(f"\n✅ Final Answer:\n{result['output']}")
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")

        print(f"{'=' * 60}\n")


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "❌ 未找到 API Key。请通过 --api-key 传入或在 .env 中设置 OPENAI_API_KEY。"
        )
        sys.exit(1)

    agent = build_agent(
        model_name=args.model,
        openai_api_key=api_key,
        openai_api_base=args.api_base,
        verbose=False,  # 我们用自定义回调来展示
    )

    callback = ReasoningCallback()

    # 单次问题模式
    if args.question:
        question = " ".join(args.question)
        print(f"❓ Question: {question}")
        result = agent.invoke(
            {"input": question},
            {"callbacks": [callback]},
        )
        print(f"\n✅ Final Answer:\n{result['output']}")
        return

    # 交互模式
    interactive_loop(agent, callback)


if __name__ == "__main__":
    main()
