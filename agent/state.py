"""
全局状态管理器
在图中每个节点间传递的中央共享状态。
"""

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document


@dataclass
class ResearchState:
    """研究代理的全局状态"""
    query: str
    mode: str = "deep_rag"  # 模式: "fast_web", "deep_rag", "local_only"
    hitl_enabled: bool = False  # 是否启用HITL回圈（默认端到端）
    test_mode: bool = False  # 是否启用测试模式（复用同一份Web搜索结果）
    web_search_cache: List[Document] = field(default_factory=list)

    research_tasks: List[str] = field(default_factory=list)
    plan_approved: bool = False

    retrieved_evidence: List[Document] = field(default_factory=list)

    reflection_iterations: int = 0
    max_reflection_iterations: int = 3

    report_draft: str = ""
    report_approved: bool = False
    final_report: str = ""

    messages: List[dict] = field(default_factory=list)  # 对话历史