"""
证据融合与总结器
将从网络检索到的文档合并成一个连贯的证据摘要。
"""

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

from agent.state import ResearchState

load_dotenv()


SYSTEM_PROMPT = """你是一名研究分析师。
你的任务是根据给定的子任务和证据，总结关键发现。

要求：
1. 提取与子任务相关的核心信息
2. 保持准确性和客观性
3. 引用来源时使用 [source_index] 格式
4. 简洁但详尽，避免无关细节"""


def fuse(state: ResearchState) -> ResearchState:
    """
    获取 state.retrieved_evidence，调用 LLM 为每个子任务进行总结。
    将结构化摘要附加到状态中。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    if not state.research_tasks or not state.retrieved_evidence:
        return state

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "qwen3-max"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0
    )

    # 将证据格式化为带索引的列表
    evidence_text = _format_evidence(state.retrieved_evidence)

    summaries = []

    for idx, task in enumerate(state.research_tasks):
        prompt = _build_fusion_prompt(task, evidence_text)

        messages: List[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)

        summaries.append({
            "task": task,
            "summary": response.content.strip()
        })

    # 将摘要添加到消息历史中
    state.messages.append({
        "role": "system",
        "content": f"=== Evidence Summaries ===\n{summaries}"
    })

    # 临时将 summaries 存储在 messages 中供 Writer 使用
    state.messages.append({
        "role": "evidence_summaries",
        "content": summaries
    })

    return state


def _format_evidence(evidence: List[object]) -> str:
    """
    将证据列表格式化为带索引的文本。

    Args:
        evidence: 证据字符串列表。

    Returns:
        格式化的证据文本。
    """
    text = "可用证据:\n\n"

    for idx, evidence_item in enumerate(evidence, start=1):
        text += f"[{idx}] {_evidence_item_to_text(evidence_item)}\n\n"

    return text


def _evidence_item_to_text(evidence_item: object) -> str:
    """
    将证据项转换为可读文本。
    """
    if isinstance(evidence_item, Document):
        url = (evidence_item.metadata or {}).get("url", "")
        source = (evidence_item.metadata or {}).get("source", "")
        header_parts = []
        if source:
            header_parts.append(f"来源: {source}")
        if url:
            header_parts.append(f"URL: {url}")
        header = " | ".join(header_parts)
        if header:
            return f"{header}\n内容: {evidence_item.page_content}"
        return evidence_item.page_content

    return str(evidence_item)


def _build_fusion_prompt(task: str, evidence_text: str) -> str:
    """
    构建证据融合提示词。

    Args:
        task: 子任务。
        evidence_text: 格式化的证据文本。

    Returns:
        提示词字符串。
    """
    prompt = f"""子任务: "{task}"

{evidence_text}

请总结与该子任务相关的关键发现。
通过 [source_index] 引用来源。
要求简洁但详尽。"""

    return prompt