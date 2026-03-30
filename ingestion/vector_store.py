"""
向量数据库接口

持久化和查询嵌入（使用 ChromaDB）。
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from uuid import uuid4

import chromadb
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    向量数据库接口，基于 ChromaDB。
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_dir: str = None
    ):
        """
        初始化向量存储。

        Args:
            collection_name: 集合名称，默认从环境变量读取。
            persist_dir: 持久化目录，默认从环境变量读取。
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION",
            "research_kb"
        )
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR",
            "./chroma_db"
        )

        # 确保持久化目录存在
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # 初始化 Chroma 客户端
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> int:
        """
        插入或更新文档到向量数据库。

        Args:
            texts: 文本列表。
            embeddings: 向量列表（与 texts 一一对应）。
            metadatas: 元数据列表（与 texts 一一对应）。
            ids: 文档 ID 列表。

        Returns:
            插入的文档数量。
        """
        if not texts:
            return 0

        if len(embeddings) != len(texts):
            raise ValueError("Embeddings size must match texts size")

        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("Metadatas size must match texts size")
        else:
            metadatas = [meta if meta else {"source": "unknown"} for meta in metadatas]

        if ids is None:
            ids = [str(uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("IDs size must match texts size")
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(texts)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        相似度搜索。

        Args:
            query_embedding: 查询向量。
            top_k: 返回的结果数量。
            filter: 元数据过滤条件。

        Returns:
            相似度最高的 Document 列表。
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter or None,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return [Document(page_content=doc, metadata=meta or {}) for doc, meta in zip(documents, metadatas)]

    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        相似度搜索（带分数）。

        Args:
            query_embedding: 查询向量。
            top_k: 返回的结果数量。
            filter: 元数据过滤条件。

        Returns:
            (Document, score) 对的列表。
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter or None,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        pairs = []
        for doc, meta, score in zip(documents, metadatas, distances):
            pairs.append((Document(page_content=doc, metadata=meta or {}), float(score)))

        return pairs

    def delete_by_ids(self, ids: List[str]) -> int:
        """
        根据 ID 删除文档。

        Args:
            ids: 文档 ID 列表。

        Returns:
            删除的文档数量。
        """
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            print(f"删除文档时出错: {e}")
            return 0

    def reset_collection(self) -> None:
        """
        重置集合（删除所有数据）。
        """
        try:
            # 删除并重新创建集合
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            print(f"重置集合时出错: {e}")

    def count(self) -> int:
        """
        获取集合中的文档数量。

        Returns:
            文档数量。
        """
        return self.collection.count()

    def get_all_documents(self) -> List[Document]:
        """
        获取集合中的所有文档。

        Returns:
            所有 Document 列表。
        """
        results = self.collection.get()
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        return [Document(page_content=doc, metadata=meta or {}) for doc, meta in zip(documents, metadatas)]