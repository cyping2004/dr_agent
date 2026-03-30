"""
嵌入器

将文本块转换为向量嵌入。
"""

import os
from typing import List, Tuple
from functools import lru_cache

from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class Embedder:
    """
    嵌入模型包装器，支持 OpenAI 和其他嵌入模型。
    """

    def __init__(self, model_name: str = None):
        """
        初始化嵌入器。

        Args:
            model_name: 嵌入模型名称，默认从环境变量读取。
        """
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-v1"
        )

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> List[float]:
        """
        嵌入单个文本（带缓存）。

        Args:
            text: 待嵌入的文本。

        Returns:
            向量嵌入列表。
        """
        # 清除缓存键中的多余空白
        text = ' '.join(text.split())
        return self._embed_texts([text])[0]

    def embed_documents(
        self,
        docs: List[Document],
        batch_size: int = 10
    ) -> List[Tuple[Document, List[float]]]:
        """
        嵌入每个文档块。

        Args:
            docs: Document 对象列表。
            batch_size: 批处理大小。

        Returns:
            (document, embedding) 对的列表。
        """
        results = []

        # 批量处理文档
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            texts = [' '.join(doc.page_content.split()) for doc in batch]

            # 调用批量嵌入
            embeddings = self._embed_texts(texts)

            # 组合结果
            for doc, embedding in zip(batch, embeddings):
                results.append((doc, embedding))

        return results

    def embed_query(self, query: str) -> List[float]:
        """
        嵌入查询文本。

        Args:
            query: 查询字符串。

        Returns:
            查询向量。
        """
        return self.embed_text(query)


# 全局单例实例
_default_embedder = None


def get_embedder() -> Embedder:
    """
    获取默认嵌入器实例。

    Returns:
        Embedder 实例。
    """
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


def embed_documents(docs: List[Document]) -> List[Tuple[Document, List[float]]]:
    """
    便捷函数：嵌入文档列表。

    Args:
        docs: Document 对象列表。

    Returns:
        (document, embedding) 对的列表。
    """
    embedder = get_embedder()
    return embedder.embed_documents(docs)


def embed_query(query: str) -> List[float]:
    """
    便捷函数：嵌入查询文本。

    Args:
        query: 查询字符串。

    Returns:
        查询向量。
    """
    embedder = get_embedder()
    return embedder.embed_query(query)