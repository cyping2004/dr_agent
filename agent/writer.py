"""
报告编写器
生成结构化的 Markdown 研究报告。
"""

import os
import re
from pathlib import Path
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
3. 语言简洁专业，尽可能多地描述你所知道的事实
4. 只输出报告正文，不要任何前言、寒暄、角色说明或解释
5. 必须以标题行开头，且只使用指定的报告结构
6. 引用事实时使用 [n] 标注，n 对应证据列表序号
7. 标题编号格式必须与模板一致"""


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

    report = _sanitize_report_text(response.content)
    report = _strip_reference_section(report)
    references, evidence_ref_map = _collect_references(state.retrieved_evidence)
    if evidence_ref_map:
        report = _remap_citations(report, evidence_ref_map)
    references_section = _build_reference_section(references)
    if references_section:
        report = f"{report}\n\n{references_section}"
    state.report_draft = report

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
        metadata = evidence_item.metadata or {}
        url = metadata.get("url", "")
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        source_type = str(metadata.get("source_type", "")).strip().lower()
        filename = str(metadata.get("filename", "")).strip()
        if not filename and source and source_type == "local":
            filename = Path(str(source)).name

        header_parts = []
        if source:
            header_parts.append(f"来源: {source}")
        if source_type:
            header_parts.append(f"类型: {source_type}")
        if filename:
            header_parts.append(f"文档: {filename}")
        if title:
            header_parts.append(f"标题: {title}")
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
    prompt = f"""请严格按以下格式输出，不要任何前言或解释：
# 深度研究报告：{query}

## 1. 执行摘要

## 2. 关键发现
（使用 2.1、2.2… 小标题组织关键发现）

## 3. 结论

## 4. 局限性与进一步研究

要求：
- 引用事实时使用 [n] 标注，n 对应证据列表序号，且从 1 开始
- 不要生成“参考来源”列表，系统会自动添加

证据如下：
{evidence_text}
"""

    return prompt


def _sanitize_report_text(text: str) -> str:
    """Strip any leading non-report text to keep a fixed report format."""
    content = (text or "").strip()
    if not content:
        return content

    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            return "\n".join(lines[idx:]).strip()

    return content


def _strip_reference_section(text: str) -> str:
    """Remove any existing reference section to avoid duplicates."""
    lines = (text or "").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"**参考来源**", "## 参考来源", "参考来源"}:
            return "\n".join(lines[:idx]).rstrip()
    return text


def _collect_references(evidence: List[object]) -> tuple[List[Dict[str, str]], List[int]]:
    """Collect unique references and map evidence indices to reference indices."""
    references: List[Dict[str, str]] = []
    key_to_index: Dict[str, int] = {}
    evidence_ref_map: List[int] = []

    for evidence_item in evidence:
        if isinstance(evidence_item, Document):
            metadata = evidence_item.metadata or {}
            title = (metadata.get("title") or "").strip()
            url = (metadata.get("url") or "").strip()
            source = str(metadata.get("source") or "").strip()
            filename = str(metadata.get("filename") or "").strip()
            if not filename and source and source not in {"web_search", "unknown"}:
                filename = Path(source).name

            source_type = str(metadata.get("source_type") or "").strip().lower()
            if not source_type:
                if url:
                    source_type = "web"
                elif filename or metadata.get("file_type"):
                    source_type = "local"
        else:
            title = ""
            url = ""
            source = ""
            filename = ""
            source_type = ""

        key = _reference_key(title, url, filename, source_type, source)
        if key not in key_to_index:
            key_to_index[key] = len(references) + 1
            references.append(
                {
                    "title": title,
                    "url": url,
                    "filename": filename,
                    "source_type": source_type,
                    "source": source,
                }
            )

        evidence_ref_map.append(key_to_index[key])

    return references, evidence_ref_map


def _reference_key(
    title: str,
    url: str,
    filename: str,
    source_type: str,
    source: str,
) -> str:
    if url:
        normalized = url.strip().rstrip("/")
        return f"url:{normalized}"
    if filename and source_type == "local":
        return f"local:{filename.lower()}"
    if filename:
        return f"file:{filename.lower()}"
    if title:
        return f"title:{title.strip().lower()}"
    if source:
        return f"source:{source.strip().lower()}"
    return "unknown"


def _remap_citations(text: str, evidence_ref_map: List[int]) -> str:
    """Remap [n] citations to the deduplicated reference indices."""
    if not text or not evidence_ref_map:
        return text

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        try:
            idx = int(raw)
        except ValueError:
            return match.group(0)
        if 1 <= idx <= len(evidence_ref_map):
            return f"[{evidence_ref_map[idx - 1]}]"
        return match.group(0)

    return re.sub(r"\[(\d+)\]", replace, text)


def _build_reference_section(references: List[Dict[str, str]]) -> str:
    """Build a reference list from unique metadata (title + url)."""
    if not references:
        return ""

    lines = []
    for idx, entry in enumerate(references, start=1):
        title = (entry.get("title") or "").strip()
        url = (entry.get("url") or "").strip()
        filename = (entry.get("filename") or "").strip()
        source_type = (entry.get("source_type") or "").strip().lower()

        if source_type == "local" and filename:
            lines.append(f"[{idx}] 本地文档: {filename}")
        elif title and url:
            lines.append(f"[{idx}] {title} ({url})")
        elif url:
            lines.append(f"[{idx}] {url}")
        elif filename:
            lines.append(f"[{idx}] 本地文档: {filename}")
        elif title:
            lines.append(f"[{idx}] {title}")
        else:
            lines.append(f"[{idx}] 未命名来源")

    return "**参考来源**\n\n" + "\n".join(lines)