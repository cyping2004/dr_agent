"""
向量数据库检索包装器
"""

import os
from typing import List

from langchain_core.documents import Document
from dotenv import load_dotenv

from agent.state import ResearchState
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

        query_embedding = embed_query(query)
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results

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

        query_embedding = embed_query(query)
        results = self.vector_store.similarity_search_with_score(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results


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
    state.retrieved_evidence = [doc.page_content for doc in retrieved_docs]

    return state