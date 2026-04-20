"""
Graph拆分模块

将研究图拆分为前半段（共享）和后半段（分叉），支持独立执行和对比测试。

前半段：Query → Planner → WebSearcher → [Cache]
后半段Fast Web：documents → Writer
后半段Deep RAG：documents → Ingest → Retrieve → Writer
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from agent.state import ResearchState
from agent.planner import plan
from agent.writer import write
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_documents
from ingestion.chunker import chunk_documents


@dataclass
class FirstHalfOutput:
    """前半段输出数据结构，符合TEST_PLAN.md定义的Schema"""
    query: str
    tasks: List[str]
    documents: List[Document]
    timestamp: str
    search_provider: str = "tavily"
    cache_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于JSON序列化"""
        return {
            "query": self.query,
            "tasks": self.tasks,
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ],
            "timestamp": self.timestamp,
            "search_provider": self.search_provider,
            "cache_version": self.cache_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirstHalfOutput":
        """从字典创建实例"""
        documents = [
            Document(page_content=doc["page_content"], metadata=doc.get("metadata", {}))
            for doc in data.get("documents", [])
        ]
        return cls(
            query=data["query"],
            tasks=data.get("tasks", []),
            documents=documents,
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            search_provider=data.get("search_provider", "tavily"),
            cache_version=data.get("cache_version", "1.0")
        )


@dataclass
class SecondHalfMetrics:
    """后半段执行指标"""
    mode: str
    top_k: int = 0
    ingest_time_ms: float = 0.0
    retrieve_time_ms: float = 0.0
    writer_time_ms: float = 0.0
    total_time_ms: float = 0.0
    original_doc_count: int = 0
    retrieved_doc_count: int = 0
    original_chunk_count: int = 0
    retrieved_chunk_count: int = 0
    original_tokens: int = 0
    retrieved_tokens: int = 0
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SecondHalfResult:
    """后半段执行结果"""
    report: str
    metrics: SecondHalfMetrics


class SplitResearchGraph:
    """
    拆分后的研究图，支持前半段和后半段独立执行

    使用方式:
    1. 执行前半段：run_first_half(query) -> FirstHalfOutput
    2. 保存/加载前半段输出（通过cache_manager）
    3. 执行后半段：run_second_half(first_half_output, mode) -> SecondHalfResult
    """

    def __init__(self, top_k: int = 5, collection_prefix: str = "test_deep_rag"):
        self.top_k = top_k
        self.collection_prefix = collection_prefix
        self.current_collection = None

    @staticmethod
    def _sanitize_collection_suffix(text: str) -> str:
        """Sanitize collection name suffix to satisfy Chroma naming rules."""
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
        safe = safe.strip("._-")
        return safe or "query"

    @staticmethod
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def build_first_half_graph(self) -> StateGraph:
        """构建前半段图：Query -> Planner -> WebSearcher"""
        graph = StateGraph(ResearchState)

        # 添加节点
        graph.add_node("planner", plan)
        graph.add_node("web_searcher", self._web_search_node)

        # 设置入口和边
        graph.set_entry_point("planner")
        graph.add_edge("planner", "web_searcher")
        graph.add_edge("web_searcher", END)

        return graph.compile()

    def run_first_half(self, query: str) -> FirstHalfOutput:
        """
        执行前半段，返回标准化的中间结果

        Args:
            query: 用户查询

        Returns:
            FirstHalfOutput: 包含query、tasks、documents等的结构化输出
        """
        # 创建初始状态
        state = ResearchState(
            query=query,
            mode="fast_web",  # 前半段模式不影响执行
            hitl_enabled=False
        )

        # 执行前半段图
        graph = self.build_first_half_graph()
        result = graph.invoke(state)

        if isinstance(result, dict):
            result = ResearchState(**result)

        # 构建输出
        return FirstHalfOutput(
            query=query,
            tasks=result.research_tasks,
            documents=result.retrieved_evidence,
            timestamp=datetime.now().isoformat(),
            search_provider="tavily",  # 从环境变量读取
            cache_version="1.0"
        )

    def run_fast_web_second_half(
        self,
        first_half_output: FirstHalfOutput
    ) -> SecondHalfResult:
        """
        执行Fast Web后半段：documents → Writer（跳过EvidenceFusion）

        Args:
            first_half_output: 前半段输出

        Returns:
            SecondHalfResult: 报告和指标
        """
        metrics = SecondHalfMetrics(mode="fast_web")
        metrics.original_doc_count = len(first_half_output.documents)
        metrics.original_tokens = sum(
            len(doc.page_content.split())
            for doc in first_half_output.documents
        )

        # 创建状态
        state = ResearchState(
            query=first_half_output.query,
            mode="fast_web",
            hitl_enabled=False,
            research_tasks=first_half_output.tasks,
            retrieved_evidence=first_half_output.documents
        )

        # 记录Writer时间
        start_time = time.time()

        # 直接执行Writer（跳过EvidenceFusion）
        result_state = write(state)

        writer_time = (time.time() - start_time) * 1000

        # 更新指标
        metrics.writer_time_ms = writer_time
        metrics.total_time_ms = writer_time
        metrics.retrieved_doc_count = metrics.original_doc_count  # Fast Web无压缩
        metrics.retrieved_tokens = metrics.original_tokens
        metrics.original_chunk_count = metrics.original_doc_count
        metrics.retrieved_chunk_count = metrics.retrieved_doc_count
        metrics.input_tokens = metrics.retrieved_tokens
        metrics.output_tokens = self._count_tokens(result_state.report_draft)

        return SecondHalfResult(
            report=result_state.report_draft,
            metrics=metrics
        )

    def run_deep_rag_second_half(
        self,
        first_half_output: FirstHalfOutput,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> SecondHalfResult:
        """
        执行Deep RAG后半段：documents → Ingest → Retrieve → Writer（跳过EvidenceFusion）

        Args:
            first_half_output: 前半段输出
            top_k: 检索Top-K文档数

        Returns:
            SecondHalfResult: 报告和指标
        """
        from ingestion.vector_store import VectorStore
        from ingestion.embedder import embed_documents
        from langchain_core.documents import Document

        metrics = SecondHalfMetrics(mode="deep_rag", top_k=top_k)
        metrics.original_doc_count = len(first_half_output.documents)

        # 创建临时集合名称
        safe_query = self._sanitize_collection_suffix(first_half_output.query)
        safe_query = safe_query[:40]
        collection_name = f"{self.collection_prefix}_{safe_query}_{int(time.time())}"
        self.current_collection = collection_name

        # ========== 阶段1: Ingest ==========
        ingest_start = time.time()

        docs = []
        for idx, doc in enumerate(first_half_output.documents):
            metadata = dict(doc.metadata or {})
            metadata.setdefault("source", "web_search")
            metadata["ingestion_session"] = collection_name
            docs.append(Document(page_content=doc.page_content, metadata=metadata))

        chunked_docs = chunk_documents(docs)
        metrics.original_chunk_count = len(chunked_docs)
        metrics.original_tokens = sum(self._count_tokens(doc.page_content) for doc in chunked_docs)

        # 嵌入并存储
        try:
            embedded_pairs = embed_documents(chunked_docs)
            texts = [doc.page_content for doc, _ in embedded_pairs]
            metadatas = [doc.metadata for doc, _ in embedded_pairs]
            embeddings = [embedding for _, embedding in embedded_pairs]

            vector_store = VectorStore(collection_name=collection_name)
            vector_store.upsert(texts, embeddings, metadatas=metadatas)
        except Exception as e:
            print(f"[警告] 向量存储失败: {e}")

        ingest_time = (time.time() - ingest_start) * 1000
        metrics.ingest_time_ms = ingest_time

        # ========== 阶段2: Retrieve ==========
        retrieve_start = time.time()

        import agent.retriever as retriever_module
        retriever_instance = retriever_module.Retriever(collection_name=collection_name)
        retrieval_mode = retriever_instance.mode

        def score_passes_threshold(score: float) -> bool:
            if score_threshold is None:
                return True
            if retrieval_mode == "dense":
                return score <= score_threshold
            return score >= score_threshold

        retrieved_docs = []
        for task in first_half_output.tasks:
            pairs = retriever_instance.retrieve_with_scores(task, top_k=top_k)
            for doc, score in pairs:
                if score_passes_threshold(score):
                    retrieved_docs.append(doc)

        retrieve_time = (time.time() - retrieve_start) * 1000
        metrics.retrieve_time_ms = retrieve_time

        metrics.retrieved_doc_count = len(retrieved_docs)
        metrics.retrieved_tokens = sum(len(doc.page_content.split()) for doc in retrieved_docs)
        metrics.input_tokens = metrics.retrieved_tokens
        metrics.retrieved_chunk_count = len(retrieved_docs)

        parent_doc_ids = {
            (doc.metadata or {}).get("parent_doc_index")
            for doc in retrieved_docs
            if doc.metadata is not None
        }
        parent_doc_ids.discard(None)
        if parent_doc_ids:
            metrics.retrieved_doc_count = len(parent_doc_ids)

        # ========== 阶段3: Writer ==========
        writer_start = time.time()

        final_state = ResearchState(
            query=first_half_output.query,
            mode="deep_rag",
            hitl_enabled=False,
            research_tasks=first_half_output.tasks,
            retrieved_evidence=retrieved_docs
        )

        result_state = write(final_state)

        writer_time = (time.time() - writer_start) * 1000
        metrics.writer_time_ms = writer_time
        metrics.output_tokens = self._count_tokens(result_state.report_draft)

        # 计算总时间
        metrics.total_time_ms = ingest_time + retrieve_time + writer_time

        # 清理临时集合（可选）
        try:
            vector_store = VectorStore(collection_name=collection_name)
            vector_store.delete_collection()
        except:
            pass

        return SecondHalfResult(
            report=result_state.report_draft,
            metrics=metrics
        )

    def _web_search_node(self, state: ResearchState) -> ResearchState:
        """网络搜索节点包装器"""
        from agent.web_searcher import search

        all_docs = []

        for task in state.research_tasks:
            print(f"[Web Search] 搜索任务: {task}")
            results = search(task, num_results=10)
            all_docs.extend(results)

        state.retrieved_evidence = all_docs
        return state
