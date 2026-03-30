"""
本地文档摄取脚本

将本地文档（PDF、TXT、MD 等）解析、分块、嵌入并存储到向量数据库。

使用方法:
    python ingest_local.py <file_or_directory> [--collection <name>]

示例:
    python ingest_local.py ./documents/sample.pdf
    python ingest_local.py ./documents/
    python ingest_local.py ./documents/ --collection my_knowledge_base
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from ingestion.parser import parse_file, parse_files
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents
from ingestion.vector_store import VectorStore
from langchain_core.documents import Document

load_dotenv()


def ingest_file(
    file_path: str,
    collection_name: str = None,
    persist_dir: str = None,
    chunk_size: int = 512,
    overlap: int = 64
) -> int:
    """
    摄取单个文件到向量数据库。

    Args:
        file_path: 文件路径。
        collection_name: 集合名称。
        persist_dir: 持久化目录。
        chunk_size: 分块大小。
        overlap: 块重叠大小。

    Returns:
        成功摄取的块数量。
    """
    print(f"\n=== 摄取文件: {file_path} ===")

    # 1. 解析文件
    print("[1/4] 解析文件...")
    try:
        docs = parse_file(file_path)
        print(f"    解析出 {len(docs)} 个文档（页/段落）")
    except Exception as e:
        print(f"    错误: 解析失败 - {e}")
        return 0

    # 2. 分块
    print("[2/4] 语义分块...")
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    print(f"    分割为 {len(chunks)} 个块")

    # 3. 嵌入
    print("[3/4] 生成嵌入向量...")
    try:
        embedded_pairs = embed_documents(chunks)
        print(f"    成功生成 {len(embedded_pairs)} 个嵌入向量")
    except Exception as e:
        print(f"    错误: 嵌入失败 - {e}")
        return 0

    # 4. 存储
    print("[4/4] 存储到向量数据库...")
    vector_store = VectorStore(collection_name=collection_name, persist_dir=persist_dir)
    texts = [doc.page_content for doc, _ in embedded_pairs]
    metadatas = [doc.metadata for doc, _ in embedded_pairs]
    embeddings = [embedding for _, embedding in embedded_pairs]

    try:
        count = vector_store.upsert(texts, embeddings, metadatas=metadatas)
        print(f"    成功存储 {count} 个文档块")
    except Exception as e:
        print(f"    错误: 存储失败 - {e}")
        return 0

    return count


def ingest_directory(
    directory: str,
    collection_name: str = None,
    persist_dir: str = None,
    chunk_size: int = 512,
    overlap: int = 64,
    recursive: bool = True
) -> int:
    """
    摄取目录中的所有支持文件到向量数据库。

    Args:
        directory: 目录路径。
        collection_name: 集合名称。
        persist_dir: 持久化目录。
        chunk_size: 分块大小。
        overlap: 块重叠大小。
        recursive: 是否递归搜索子目录。

    Returns:
        成功摄取的块总数。
    """
    path = Path(directory)

    if not path.exists() or not path.is_dir():
        print(f"错误: 目录不存在: {directory}")
        return 0

    # 查找所有支持的文件
    supported_extensions = {'.pdf', '.txt', '.md'}
    files = []

    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'

    for file_path in path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(str(file_path))

    if not files:
        print(f"警告: 目录中没有找到支持的文件: {directory}")
        return 0

    print(f"\n=== 摄取目录: {directory} ===")
    print(f"找到 {len(files)} 个文件")

    total_chunks = 0
    successful_files = 0

    # 批量处理文件
    all_chunks = []
    for file_path in files:
        print(f"\n处理文件: {file_path}")

        try:
            # 解析和分块
            docs = parse_file(file_path)
            chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)

            all_chunks.extend(chunks)
            total_chunks += len(chunks)
            successful_files += 1

            print(f"  -> 解析出 {len(docs)} 个文档，分割为 {len(chunks)} 个块")

        except Exception as e:
            print(f"  -> 错误: {e}")
            continue

    if not all_chunks:
        print("\n错误: 没有成功处理任何文件")
        return 0

    # 批量嵌入和存储
    print(f"\n总共分割为 {total_chunks} 个块")
    print(f"[1/2] 生成嵌入向量...")

    try:
        embedded_pairs = embed_documents(all_chunks)
        print(f"    成功生成 {len(embedded_pairs)} 个嵌入向量")
    except Exception as e:
        print(f"    错误: 嵌入失败 - {e}")
        return 0

    print(f"[2/2] 存储到向量数据库...")
    vector_store = VectorStore(collection_name=collection_name, persist_dir=persist_dir)
    texts = [doc.page_content for doc, _ in embedded_pairs]
    metadatas = [doc.metadata for doc, _ in embedded_pairs]
    embeddings = [embedding for _, embedding in embedded_pairs]

    try:
        count = vector_store.upsert(texts, embeddings, metadatas=metadatas)
        print(f"    成功存储 {count} 个文档块")
    except Exception as e:
        print(f"    错误: 存储失败 - {e}")
        return 0

    return count


def query_vector_store(
    query: str,
    collection_name: str = None,
    persist_dir: str = None,
    top_k: int = 5
):
    """
    查询向量数据库。

    Args:
        query: 查询字符串。
        collection_name: 集合名称。
        persist_dir: 持久化目录。
        top_k: 返回结果数量。
    """
    print(f"\n=== 查询向量数据库 ===")
    print(f"查询: {query}")

    vector_store = VectorStore(collection_name=collection_name, persist_dir=persist_dir)

    results = vector_store.similarity_search_with_score(query, top_k=top_k)

    print(f"\n找到 {len(results)} 个结果:\n")

    for idx, (doc, score) in enumerate(results, 1):
        print(f"[{idx}] 相似度: {score:.4f}")
        print(f"    来源: {doc.metadata.get('source', 'N/A')}")
        if 'page' in doc.metadata:
            print(f"    页码: {doc.metadata['page']}")
        if 'filename' in doc.metadata:
            print(f"    文件: {doc.metadata['filename']}")
        print(f"    内容: {doc.page_content[:200]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="本地文档摄取工具 - 将文档解析、分块、嵌入并存储到向量数据库"
    )

    parser.add_argument(
        "path",
        help="文件或目录路径"
    )

    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "research_kb"),
        help="向量数据库集合名称（默认: research_kb）"
    )

    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="向量数据库持久化目录（默认: ./chroma_db）"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="分块大小（默认: 512）"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="块重叠大小（默认: 64）"
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="目录模式下不递归搜索子目录"
    )

    parser.add_argument(
        "--query",
        help="查询模式：查询向量数据库而不进行摄取"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="查询模式下返回的结果数量（默认: 5）"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置（清空）集合"
    )

    args = parser.parse_args()

    # 重置模式
    if args.reset:
        print(f"重置集合: {args.collection}")
        vector_store = VectorStore(collection_name=args.collection, persist_dir=args.persist_dir)
        vector_store.reset_collection()
        print(f"集合 {args.collection} 已清空")
        return 0

    # 查询模式
    if args.query:
        query_vector_store(
            query=args.query,
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            top_k=args.top_k
        )
        return 0

    # 摄取模式
    path = Path(args.path)

    if path.is_file():
        # 摄取单个文件
        count = ingest_file(
            file_path=str(path),
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
    elif path.is_dir():
        # 摄取目录
        count = ingest_directory(
            directory=str(path),
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recursive=not args.no_recursive
        )
    else:
        print(f"错误: 路径不存在: {args.path}")
        return 1

    # 汇总
    print(f"\n=== 完成 ===")
    print(f"集合: {args.collection}")
    print(f"持久化目录: {args.persist_dir}")
    print(f"成功摄取: {count} 个文档块")

    return 0


if __name__ == "__main__":
    sys.exit(main())