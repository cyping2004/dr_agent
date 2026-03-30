"""
规划器
将用户查询分解为一系列具体的研究子任务。
"""

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv

from agent.state import ResearchState

load_dotenv()


def plan(state: ResearchState) -> ResearchState:
    """
    使用用户查询 + 可选的用户反馈调用 LLM。
    更新 state.research_tasks。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "qwen-max"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0
    )

    # 构建提示
    prompt = _build_planning_prompt(state)

    # 调用 LLM
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    # 解析响应
    tasks = _parse_planning_response(response.content)

    # 更新状态
    state.research_tasks = tasks

    return state


SYSTEM_PROMPT = """你是一个研究规划助理。
你的任务是将用户的查询分解为 3–7 个具体的、可搜索的子问题。

要求：
1. 每个子问题应该是独立的，能够通过搜索获得答案
2. 子问题之间应该互补，覆盖原始查询的各个方面
3. 保持精确，避免重叠
4. 返回格式为编号列表"""


def _build_planning_prompt(state: ResearchState) -> str:
    """构建规划提示词"""
    prompt = f"""给定用户查询: "{state.query}"
"""

    # 如果有用户反馈（在 messages 中查找）
    if state.messages:
        feedback = state.messages[-1].get("content", "")
        if feedback and "feedback" in feedback.lower():
            prompt += f"""
用户对先前计划的反馈: {feedback}
请根据反馈调整你的计划。
"""

    prompt += """
请将此查询分解为 3–7 个具体的、可搜索的子问题。
返回一个编号列表。"""

    return prompt


def _parse_planning_response(response: str) -> List[str]:
    """
    解析规划器响应，提取子任务列表。

    Args:
        response: LLM 响应文本。

    Returns:
        子任务列表。
    """
    tasks = []

    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 移除编号前缀 (如 "1.", "1)", "- ", "* ")
        import re
        line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)

        if line:
            tasks.append(line)

    # 如果解析失败，尝试将整个响应作为单个任务
    if not tasks and response.strip():
        tasks = [response.strip()]

    return tasks