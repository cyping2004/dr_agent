"""
语义分块器

将文档分割成语义连贯的块以便嵌入。
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    overlap: int = 64,
    chunk_by: str = "char"
) -> List[Document]:
    """
    应用具有语义感知的递归字符分割。
    返回保留元数据的较小 Document 块。

    Args:
        docs: 原始 Document 列表。
        chunk_size: 每个块的最大字符数。
        overlap: 块之间的重叠字符数。
        chunk_by: 分块方式，"char" 或 "word"。

    Returns:
        分块后的 Document 列表。
    """
    if not docs:
        return []

    # 配置文本分割器
    separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=separators,
        keep_separator=False,
    )

    # 对每个文档进行分块
    all_chunks = []
    for doc_idx, doc in enumerate(docs):
        text = doc.page_content
        metadata = doc.metadata.copy()

        # 使用分割器进行分块
        chunks = splitter.split_text(text)

        # 为每个块创建 Document 对象
        for chunk_idx, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_idx,
                "parent_doc_index": doc_idx,
                "chunk_count": len(chunks),
            })

            chunk_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk_doc)

    return all_chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64
) -> List[Document]:
    """
    对单个文档进行分块。

    Args:
        doc: 单个 Document 对象。
        chunk_size: 每个块的最大字符数。
        overlap: 块之间的重叠字符数。

    Returns:
        分块后的 Document 列表。
    """
    return chunk_documents([doc], chunk_size, overlap)


def merge_chunks(chunks: List[Document]) -> List[Document]:
    """
    将来自同一父文档的块合并回原始文档结构。

    Args:
        chunks: 分块后的 Document 列表。

    Returns:
        合并后的 Document 列表。
    """
    # 按 parent_doc_index 和 chunk_index 分组
    groups = {}
    for chunk in chunks:
        parent_idx = chunk.metadata.get("parent_doc_index", 0)
        if parent_idx not in groups:
            groups[parent_idx] = []
        groups[parent_idx].append(chunk)

    # 合并每个组
    merged_docs = []
    for parent_idx, group in sorted(groups.items()):
        # 按 chunk_index 排序
        group.sort(key=lambda x: x.metadata.get("chunk_index", 0))

        # 合并内容
        content = " ".join(chunk.page_content for chunk in group)

        # 保留原始元数据
        metadata = group[0].metadata.copy()
        metadata.pop("chunk_index", None)
        metadata.pop("parent_doc_index", None)
        metadata.pop("chunk_count", None)

        merged_doc = Document(
            page_content=content,
            metadata=metadata
        )
        merged_docs.append(merged_doc)

    return merged_docs