"""
嵌入器

将文本块转换为向量嵌入。
"""

import os
from typing import List, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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

        self.local_model_path = os.getenv("EMBEDDING_MODEL_PATH") or os.getenv("EMBEDDING_LOCAL_PATH")
        self.local_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.local_batch_size = _get_int_env("EMBEDDING_LOCAL_BATCH_SIZE", 32)
        self.local_normalize = _get_bool_env("EMBEDDING_NORMALIZE", True)
        self.local_model = self._load_local_model(self.local_model_path) if self.local_model_path else None

        self.parallel_enabled = _get_bool_env("EMBEDDING_PARALLEL", True)
        self.max_workers = _get_int_env("EMBEDDING_MAX_WORKERS", 2)

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def _load_local_model(self, model_path: str) -> Optional["SentenceTransformer"]:
        if SentenceTransformer is None:
            print("[Embedder] sentence-transformers not installed; using API embeddings.")
            return None
        try:
            return SentenceTransformer(
                model_path,
                trust_remote_code=True,
                device=self.local_device,
            )
        except Exception as e:
            print(f"[Embedder] Failed to load local embedding model: {e}. Using API embeddings.")
            return None

    def _embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if self.local_model is not None:
            try:
                local_batch_size = batch_size or self.local_batch_size
                embeddings = self.local_model.encode(
                    texts,
                    batch_size=local_batch_size,
                    normalize_embeddings=self.local_normalize,
                    show_progress_bar=False,
                )
                return [embedding.tolist() for embedding in embeddings]
            except Exception as e:
                print(f"[Embedder] Local embedding failed: {e}. Falling back to API.")

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
        return self._embed_texts([text], batch_size=1)[0]

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

        if batch_size <= 0:
            batch_size = 1

        batches: List[Tuple[int, List[Document], List[str]]] = []
        for batch_index, i in enumerate(range(0, len(docs), batch_size)):
            batch = docs[i:i + batch_size]
            texts = [' '.join(doc.page_content.split()) for doc in batch]
            batches.append((batch_index, batch, texts))

        if not self.parallel_enabled or len(batches) == 1:
            for _, batch, texts in batches:
                embeddings = self._embed_texts(texts, batch_size=batch_size)
                for doc, embedding in zip(batch, embeddings):
                    results.append((doc, embedding))
            return results

        embeddings_by_index: dict[int, Tuple[List[Document], List[List[float]]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._embed_texts, texts, batch_size): (batch_index, batch)
                for batch_index, batch, texts in batches
            }
            for future in as_completed(future_map):
                batch_index, batch = future_map[future]
                embeddings = future.result()
                embeddings_by_index[batch_index] = (batch, embeddings)

        for batch_index in sorted(embeddings_by_index):
            batch, embeddings = embeddings_by_index[batch_index]
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