"""
Agent 构建 — 组装 LLM + 计算器工具的 ReAct Agent。
"""

from typing import Optional

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from tools import calculator


_PROMPT = PromptTemplate.from_template(
    """你是一个数学解题助手，擅长把复杂问题拆解成一步步计算。
你有计算器工具可用，遇到任何需要计算的地方，必须调用计算器工具，不要自己心算。
请尽量将计算分解成小的中间步骤，逐步求解，并用中文给出最终答案。

{input}
{agent_scratchpad}"""
)


def build_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    verbose: bool = False,
) -> AgentExecutor:
    """构建 ReAct Agent。

    Args:
        model_name: LLM 模型名称。
        temperature: 生成温度，数学题推荐 0.0。
        openai_api_key: OpenAI API Key，缺省从环境变量读取。
        openai_api_base: 自定义 API 地址（如本地 vLLM / Ollama）。
        verbose: 是否打印中间步骤。

    Returns:
        配置好的 AgentExecutor。
    """
    llm_kwargs = {}
    if openai_api_key:
        llm_kwargs["api_key"] = openai_api_key
    if openai_api_base:
        llm_kwargs["base_url"] = openai_api_base

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        **llm_kwargs,
    )

    tools = [calculator]

    agent = create_tool_calling_agent(llm, tools, prompt=_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
    )
