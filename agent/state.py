"""
全局状态管理器
在图中每个节点间传递的中央共享状态。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ResearchState:
    """研究代理的全局状态"""
    query: str
    mode: str = "fast_web"  # 模式: "fast_web", "deep_rag", "local_only"
    auto_approve: bool = False  # 是否跳过HITL批准

    research_tasks: List[str] = field(default_factory=list)
    plan_approved: bool = False

    retrieved_evidence: List[str] = field(default_factory=list)

    reflection_iterations: int = 0
    max_reflection_iterations: int = 3

    report_draft: str = ""
    report_approved: bool = False
    final_report: str = ""

    messages: List[dict] = field(default_factory=list)  # 对话历史