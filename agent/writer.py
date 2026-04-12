"""
报告编写器
生成结构化的 Markdown 研究报告。
"""

import os
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

from agent.state import ResearchState

load_dotenv()


SYSTEM_PROMPT = """你是一名专业的研究报告撰写者。
你的任务是根据研究查询和证据摘要，撰写一份结构化的 Markdown 研究报告。

报告结构要求：
1. 执行摘要 - 简要概述研究发现
2. 关键发现 - 按子任务分类的核心发现
3. 结论 - 总结性结论
4. 局限性与进一步研究 - 如有需要

写作要求：
1. 使用 Markdown 格式
2. 保持客观性和准确性
3. 语言简洁专业，尽可能多地描述你所知道的事实"""


def write(state: ResearchState) -> ResearchState:
    """
    从证据摘要生成 state.report_draft。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "qwen3-max"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0
    )

    # 获取证据摘要
    evidence_summaries = _extract_evidence_summaries(state)

    # if not evidence_summaries:
    #     # 如果没有摘要，使用原始证据
    #     evidence_text = _format_evidence(state.retrieved_evidence)
    # else:
    #     evidence_text = _format_summaries(evidence_summaries)

    per_doc_max_chars: Optional[int] = None
    if state.mode == "fast_web":
        # Cap per-document length to avoid oversized prompts in Fast Web.
        per_doc_max_chars = _get_fast_web_doc_max_chars()

    # 直接使用原始证据
    evidence_text = _format_evidence(state.retrieved_evidence, per_doc_max_chars)

    prompt = _build_writer_prompt(state.query, evidence_text)

    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)

    state.report_draft = response.content.strip()

    return state


def _extract_evidence_summaries(state: ResearchState) -> List[Dict]:
    """
    从状态中提取证据摘要。

    Args:
        state: 当前研究状态。

    Returns:
        证据摘要列表。
    """
    summaries = []

    for msg in state.messages:
        if msg.get("role") == "evidence_summaries":
            content = msg.get("content", [])
            if isinstance(content, list):
                summaries.extend(content)
            else:
                summaries.append(content)

    return summaries


def _format_evidence(
    evidence: List[object],
    per_doc_max_chars: Optional[int] = None
) -> str:
    """
    格式化原始证据。

    Args:
        evidence: 证据字符串列表。

    Returns:
        格式化的证据文本。
    """
    if not evidence:
        return "暂无证据"

    text = "收集到的证据:\n\n"

    for idx, evidence_item in enumerate(evidence, start=1):
        text += f"[{idx}] {_evidence_item_to_text(evidence_item, per_doc_max_chars)}\n\n"

    return text


def _evidence_item_to_text(
    evidence_item: object,
    max_chars: Optional[int] = None
) -> str:
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
        content = evidence_item.page_content
        if max_chars:
            content = _truncate_text(content, max_chars)
        if header:
            return f"{header}\n内容: {content}"
        return content

    return str(evidence_item)


def _format_summaries(summaries: List[Dict]) -> str:
    """
    格式化证据摘要。

    Args:
        summaries: 摘要字典列表。

    Returns:
        格式化的摘要文本。
    """
    if not summaries:
        return "暂无证据摘要"

    text = "证据摘要:\n\n"

    for idx, summary in enumerate(summaries, start=1):
        task = summary.get("task", "")
        summary_text = summary.get("summary", "")

        text += f"### 任务 {idx}: {task}\n\n"
        text += f"{summary_text}\n\n"

    return text


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _get_fast_web_doc_max_chars() -> int:
    value = os.getenv("FAST_WEB_DOC_MAX_CHARS")
    if value is None:
        return 1000
    try:
        parsed = int(value)
    except ValueError:
        return 1000
    return parsed if parsed > 0 else 1000


def _build_writer_prompt(query: str, evidence_text: str) -> str:
    """
    构建报告编写提示词。

    Args:
        query: 原始查询。
        evidence_text: 格式化的证据文本。

    Returns:
        提示词字符串。
    """
    prompt = f"""# 深度研究报告: {query}

{evidence_text}

请根据以上证据，撰写一份结构化的 Markdown 研究报告。"""

    return prompt