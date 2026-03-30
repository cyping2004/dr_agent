"""
研究代理编排图
使用 LangGraph 将所有节点连接在一起。
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import ResearchState
from agent.planner import plan
from agent.router import route
from agent.retriever import retrieve
from agent.web_searcher import search
from agent.evidence_fusion import fuse
from agent.writer import write
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_documents


def build_graph() -> StateGraph:
    """
    构建研究代理的 LangGraph。

    支持三种运行模式:
    1. fast_web: Planner -> WebSearcher -> EvidenceFusion -> Writer
    2. deep_rag: Planner -> WebSearcher -> Ingestion -> Retriever -> EvidenceFusion -> Writer
    3. local_only: Planner -> Retriever -> EvidenceFusion -> Writer

    Returns:
        编译后的 StateGraph。
    """
    graph = StateGraph(ResearchState)

    # 添加节点
    graph.add_node("planner", plan)
    graph.add_node("router", _router_node)
    graph.add_node("web_searcher", _web_search_node)
    graph.add_node("ingester", _ingestion_node)
    graph.add_node("retriever", _retriever_node)
    graph.add_node("evidence_fusion", fuse)
    graph.add_node("writer", write)

    # 设置入口点
    graph.set_entry_point("planner")

    # 边：Planner -> Router
    graph.add_edge("planner", "router")

    # 条件边：Router 根据模式决定下一步
    graph.add_conditional_edges(
        "router",
        _route_based_on_mode,
        {
            "web_searcher": "web_searcher",
            "retriever": "retriever",
            "deep_rag": "web_searcher",  # Deep RAG 先进行 Web 搜索
        }
    )

    # Web Searcher 边
    graph.add_edge("web_searcher", "evidence_fusion")  # Fast Web 模式
    graph.add_conditional_edges(
        "web_searcher",
        lambda s: "ingester" if s.mode == "deep_rag" else "evidence_fusion",
        {
            "ingester": "ingester",
            "evidence_fusion": "evidence_fusion"
        }
    )

    # Ingester 边（Deep RAG 模式）
    graph.add_edge("ingester", "retriever")

    # Retriever 边
    graph.add_edge("retriever", "evidence_fusion")

    # Evidence Fusion -> Writer
    graph.add_edge("evidence_fusion", "writer")

    # Writer -> END
    graph.add_edge("writer", END)

    return graph.compile()


def _router_node(state: ResearchState) -> ResearchState:
    """
    路由器节点：根据模式决定执行路径。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    return state


def _route_based_on_mode(state: ResearchState) -> Literal["web_searcher", "retriever", "deep_rag"]:
    """
    根据运行模式路由到下一个节点。

    Args:
        state: 当前研究状态。

    Returns:
        下一个节点的名称。
    """
    mode = state.mode

    if mode == "fast_web":
        return "web_searcher"
    elif mode == "local_only":
        return "retriever"
    elif mode == "deep_rag":
        return "deep_rag"
    else:
        return "web_searcher"


def _web_search_node(state: ResearchState) -> ResearchState:
    """
    网络搜索节点：为所有研究任务执行搜索。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    all_evidence = []

    for task in state.research_tasks:
        print(f"[Web Search] 搜索任务: {task}")
        results = search(task, num_results=3)

        for result in results:
            content = result.page_content
            url = result.metadata.get("url", "")

            # 格式化为带来源的证据
            evidence_item = f"来源: {url}\n内容: {content}"
            all_evidence.append(evidence_item)

    state.retrieved_evidence = all_evidence

    return state


def _ingestion_node(state: ResearchState) -> ResearchState:
    """
    摄取节点：将网络搜索结果存储到向量数据库（Deep RAG 模式）。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    print("[Ingestion] 将网络搜索结果摄入向量数据库...")

    # 将文本证据转换为 Document 对象
    from langchain_core.documents import Document

    docs = []
    for idx, evidence in enumerate(state.retrieved_evidence):
        # 解析证据格式：来源: {url}\n内容: {content}
        lines = evidence.split('\n', 1)
        url = ""
        content = evidence

        if len(lines) > 1 and lines[0].startswith("来源:"):
            url = lines[0].replace("来源:", "").strip()
            content = lines[1].replace("内容:", "").strip()

        doc = Document(
            page_content=content,
            metadata={
                "source": "web_search",
                "url": url,
                "ingestion_session": f"{state.query[:20]}_{idx}",
            }
        )
        docs.append(doc)

    # 存储到向量数据库
    try:
        embedded_pairs = embed_documents(docs)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]

        vector_store = VectorStore(collection_name="temp_web_results")
        vector_store.upsert(texts, embeddings, metadatas=metadatas)
        print(f"[Ingestion] 已存储 {len(docs)} 个文档到向量数据库")
    except Exception as e:
        print(f"[Ingestion] 警告: 存储到向量数据库失败: {e}")

    return state


def _retriever_node(state: ResearchState) -> ResearchState:
    """
    检索器节点：从向量数据库检索相关文档。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    # 如果是 local_only 模式，使用主集合
    # 如果是 deep_rag 模式，使用临时 Web 结果集合
    collection_name = "temp_web_results" if state.mode == "deep_rag" else None

    retriever_node = retrieve

    # 临时覆盖集合名称
    import agent.retriever as retriever_module
    original_get_retriever = retriever_module.get_retriever

    if state.mode == "deep_rag":
        def get_temp_retriever():
            return retriever_module.Retriever(collection_name=collection_name)
        retriever_module.get_retriever = get_temp_retriever

    state = retriever_node(state)

    # 恢复原始函数
    retriever_module.get_retriever = original_get_retriever

    print(f"[Retriever] 从 {collection_name or '本地知识库'} 检索到 {len(state.retrieved_evidence)} 条证据")

    return state