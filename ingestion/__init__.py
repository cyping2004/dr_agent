"""
文档摄取模块

负责解析、分块、嵌入和存储文档到向量数据库。
"""

from .parser import parse_file
from .chunker import chunk_documents
from .embedder import embed_documents
from .vector_store import VectorStore

__all__ = [
    "parse_file",
    "chunk_documents",
    "embed_documents",
    "VectorStore",
]