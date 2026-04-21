"""
路由器

根据运行模式决定检索路径和数据流向。
"""

from typing import Literal

from agent.state import ResearchState


def route(state: ResearchState) -> Literal["web_searcher", "retriever", "deep_rag"]:
    """
    根据模式决定数据流向。

    Args:
        state: 当前研究状态。

    Returns:
        下一个节点的名称。
        - "web_searcher": 仅进行网络搜索（Fast Web 模式）
        - "retriever": 仅进行本地检索（Local Only 模式）
        - "deep_rag": 网络搜索 + 向量库检索（Deep RAG 模式）

    路径说明:
        1. Fast Web (Baseline):
            - 路径: Planner -> WebSearcher -> EvidenceFusion -> Writer
            - 数据: 搜索引擎结果直接作为上下文

        2. Deep RAG (Context Compression):
            - 路径: Planner -> WebSearcher -> Ingester -> Retriever -> EvidenceFusion -> Writer
            - 数据: Web 结果先入库，再通过相似度匹配提取精华

        3. Hybrid Deep RAG:
            - 路径: LocalIngestion -> Planner -> WebSearcher -> Ingester -> Retriever -> Writer
            - 数据: 本地文档先入库，再与 Web 结果汇合检索

        4. Local Only:
            - 路径: Planner -> Retriever -> EvidenceFusion -> Writer
            - 数据: 仅从本地向量库检索
    """
    mode = state.mode

    if mode == "fast_web":
        # 极速网络模式：直接使用网络搜索结果
        return "web_searcher"

    elif mode == "local_only":
        # 纯本地模式：仅从向量库检索
        return "retriever"

    elif mode in {"deep_rag", "hybrid_deep_rag"}:
        # 深度 RAG 模式：网络搜索 + 向量库压缩
        return "deep_rag"

    else:
        # 默认返回极速网络模式
        return "web_searcher"


def should_use_web_search(state: ResearchState) -> bool:
    """
    判断是否需要使用网络搜索。

    Args:
        state: 当前研究状态。

    Returns:
        True 如果需要网络搜索。
    """
    return state.mode in ["fast_web", "deep_rag", "hybrid_deep_rag"]


def should_use_local_retrieval(state: ResearchState) -> bool:
    """
    判断是否需要使用本地检索。

    Args:
        state: 当前研究状态。

    Returns:
        True 如果需要本地检索。
    """
    return state.mode in ["local_only", "deep_rag", "hybrid_deep_rag"]


def should_ingest_web_results(state: ResearchState) -> bool:
    """
    判断是否需要将网络搜索结果摄入向量库。

    仅在 Deep RAG 模式下返回 True。

    Args:
        state: 当前研究状态。

    Returns:
        True 如果需要摄入网络结果。
    """
    return state.mode in {"deep_rag", "hybrid_deep_rag"}