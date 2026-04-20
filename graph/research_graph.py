"""
研究代理编排图
使用 LangGraph 将所有节点连接在一起。
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import ResearchState
from agent.planner import plan
from agent.web_searcher import search
from agent.writer import write
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_documents
from ingestion.chunker import chunk_documents


def build_graph() -> StateGraph:
    """
    构建研究代理的 LangGraph。

    支持两种运行模式:
    1. fast_web: Planner -> WebSearcher -> Writer
    2. deep_rag: Planner -> WebSearcher -> Ingestion -> Retriever -> Writer

    Returns:
        编译后的 StateGraph。
    """
    graph = StateGraph(ResearchState)

    # 添加节点
    graph.add_node("planner", plan)
    graph.add_node("web_searcher", _web_search_node)
    graph.add_node("ingester", _ingestion_node)
    graph.add_node("retriever", _retriever_node)
    graph.add_node("writer", write)

    # 设置入口点
    graph.set_entry_point("planner")

    # 边：Planner -> WebSearcher
    graph.add_edge("planner", "web_searcher")

    # 条件边：WebSearcher 根据模式决定下一步
    graph.add_conditional_edges(
        "web_searcher",
        _route_after_web_search,
        {
            "writer": "writer",
            "ingester": "ingester",
        }
    )

    # Ingester 边（Deep RAG 模式）
    graph.add_edge("ingester", "retriever")

    # Retriever 边
    graph.add_edge("retriever", "writer")

    # Writer -> END
    graph.add_edge("writer", END)

    return graph.compile()


def _route_after_web_search(state: ResearchState) -> Literal["writer", "ingester"]:
    """
    WebSearcher 后的路由逻辑。

    Args:
        state: 当前研究状态。

    Returns:
        下一个节点的名称。
    """
    if state.mode == "deep_rag":
        return "ingester"
    return "writer"


def _web_search_node(state: ResearchState) -> ResearchState:
    """
    网络搜索节点：为所有研究任务执行搜索。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的 ResearchState。
    """
    if state.test_mode and state.web_search_cache:
        state.retrieved_evidence = list(state.web_search_cache)
        return state

    all_docs = []

    for task in state.research_tasks:
        print(f"[Web Search] 搜索任务: {task}")
        results = search(task, num_results=10)
        all_docs.extend(results)

    state.retrieved_evidence = all_docs

    if state.test_mode:
        state.web_search_cache = list(all_docs)

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

    # 将 Web 结果补充 session 元数据后摄取
    from langchain_core.documents import Document

    docs = []
    for idx, evidence in enumerate(state.retrieved_evidence):
        if isinstance(evidence, Document):
            metadata = dict(evidence.metadata or {})
            metadata.setdefault("source", "web_search")
            metadata["ingestion_session"] = "temp_web_results"
            docs.append(Document(page_content=evidence.page_content, metadata=metadata))
        else:
            docs.append(
                Document(
                    page_content=str(evidence),
                    metadata={
                        "source": "web_search",
                        "ingestion_session": "temp_web_results",
                    }
                )
            )

    chunked_docs = chunk_documents(docs)

    # 存储到向量数据库
    try:
        embedded_pairs = embed_documents(chunked_docs)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]

        vector_store = VectorStore(collection_name="temp_web_results")
        vector_store.upsert(texts, embeddings, metadatas=metadatas)
        print(f"[Ingestion] 已存储 {len(chunked_docs)} 个文档到向量数据库")
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
    collection_name = "temp_web_results"

    import agent.retriever as retriever_module
    retriever = retriever_module.Retriever(collection_name=collection_name)

    retrieved_docs = []
    for task in state.research_tasks:
        retrieved_docs.extend(retriever.retrieve(task))

    state.retrieved_evidence = retrieved_docs

    print(f"[Retriever] 从 {collection_name} 检索到 {len(retrieved_docs)} 条证据")

    return state