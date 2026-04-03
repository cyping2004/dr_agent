"""
向量数据库检索包装器
"""

import os
from typing import List

from langchain_core.documents import Document
from dotenv import load_dotenv

from agent.state import ResearchState
from agent.sparse_bm25 import BM25Index, doc_key
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_query

load_dotenv()


class Retriever:
    """
    向量数据库检索器。
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_dir: str = None
    ):
        """
        初始化检索器。

        Args:
            collection_name: 集合名称。
            persist_dir: 持久化目录。
        """
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir
        )
        self.top_k = int(os.getenv("LOCAL_RETRIEVAL_TOP_K", "5"))

        # Retrieval mode: dense | bm25 | hybrid_rrf
        self.mode = os.getenv("RETRIEVAL_MODE", "hybrid_rrf").strip().lower()

        # BM25 lazy index
        self._bm25: BM25Index | None = None
        self._bm25_doc_count: int | None = None

        # RRF constant (typical defaults are 60)
        self.rrf_k = int(os.getenv("RRF_K", "60"))

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        从向量数据库检索相关文档。

        Args:
            query: 查询字符串。
            top_k: 返回结果数量，默认使用配置值。

        Returns:
            相关文档列表。
        """
        if top_k is None:
            top_k = self.top_k

        mode = self.mode
        if mode == "dense":
            query_embedding = embed_query(query)
            return self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        if mode == "bm25":
            return self._retrieve_bm25(query=query, top_k=top_k)
        if mode in {"hybrid", "hybrid_rrf", "rrf"}:
            return self._retrieve_hybrid_rrf(query=query, top_k=top_k)

        # fallback
        query_embedding = embed_query(query)
        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def retrieve_with_scores(self, query: str, top_k: int = None) -> List[tuple[Document, float]]:
        """
        从向量数据库检索相关文档（带相似度分数）。

        Args:
            query: 查询字符串。
            top_k: 返回结果数量。

        Returns:
            (Document, score) 对的列表。
        """
        if top_k is None:
            top_k = self.top_k

        mode = self.mode
        if mode == "dense":
            query_embedding = embed_query(query)
            return self.vector_store.similarity_search_with_score(
                query_embedding=query_embedding,
                top_k=top_k,
            )
        if mode == "bm25":
            return self._retrieve_bm25_with_scores(query=query, top_k=top_k)
        if mode in {"hybrid", "hybrid_rrf", "rrf"}:
            return self._retrieve_hybrid_rrf_with_scores(query=query, top_k=top_k)

        query_embedding = embed_query(query)
        return self.vector_store.similarity_search_with_score(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def _ensure_bm25(self) -> BM25Index:
        """Build/refresh BM25 index from VectorStore when needed."""

        current_count = self.vector_store.count()
        if self._bm25 is None or self._bm25_doc_count != current_count:
            docs = self.vector_store.get_all_documents()
            bm25 = BM25Index()
            bm25.build(docs)
            self._bm25 = bm25
            self._bm25_doc_count = current_count
        return self._bm25

    def _retrieve_bm25(self, query: str, top_k: int) -> List[Document]:
        bm25 = self._ensure_bm25()
        pairs = bm25.search(query, top_k=top_k)
        return [d for d, _ in pairs]

    def _retrieve_bm25_with_scores(self, query: str, top_k: int) -> List[tuple[Document, float]]:
        bm25 = self._ensure_bm25()
        return bm25.search(query, top_k=top_k)

    def _retrieve_hybrid_rrf(self, query: str, top_k: int) -> List[Document]:
        pairs = self._retrieve_hybrid_rrf_with_scores(query=query, top_k=top_k)
        return [d for d, _ in pairs]

    def _retrieve_hybrid_rrf_with_scores(self, query: str, top_k: int) -> List[tuple[Document, float]]:
        # Dense ranking
        query_embedding = embed_query(query)
        fetch_k = top_k * 4  # 或者从环境变量配置
        dense_pairs = self.vector_store.similarity_search_with_score(
            query_embedding=query_embedding,
            top_k=fetch_k,
        )
        # Chroma distances: smaller is better; keep returned order as rank
        dense_ranked_docs = [d for d, _ in dense_pairs]

        # Sparse ranking
        sparse_pairs = self._ensure_bm25().search(query, top_k=fetch_k)
        sparse_ranked_docs = [d for d, _ in sparse_pairs]

        fused = rrf_fuse(
            ranked_lists=[dense_ranked_docs, sparse_ranked_docs],
            k=self.rrf_k,
        )

        return fused[: max(top_k, 1)]


def rrf_fuse(ranked_lists: List[List[Document]], k: int = 60) -> List[tuple[Document, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Score(doc) = sum_i 1 / (k + rank_i(doc)) where rank starts at 1.
    """

    if k < 0:
        k = 1

    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for ranked in ranked_lists:
        for idx, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (float(k) + float(idx))

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs_by_key[key], float(score)) for key, score in ordered]


# 全局单例实例
_default_retriever = None


def get_retriever() -> Retriever:
    """
    获取默认检索器实例。

    Returns:
        Retriever 实例。
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever


def retrieve(state: ResearchState) -> ResearchState:
    """
    根据研究任务检索相关文档。

    对于每个研究任务，检索相关的本地文档。

    Args:
        state: 当前研究状态。

    Returns:
        更新后的研究状态，包含检索到的证据。
    """
    retriever = get_retriever()
    retrieved_docs = []

    # 根据研究任务检索
    for task in state.research_tasks:
        docs = retriever.retrieve(task)
        retrieved_docs.extend(docs)

    # 更新状态
    state.retrieved_evidence = retrieved_docs

    return state