"""
多模态解析器

将原始本地文件（PDF, DOCX, 图片）解析为结构化文本。
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document


def parse_file(file_path: str) -> List[Document]:
    """
    检测文件类型，分派到相应的子解析器。
    返回一个包含 .page_content 和 .metadata 的 Document 对象列表。

    Args:
        file_path: 文件路径。

    Returns:
        Document 对象列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不支持的文件类型。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_ext = path.suffix.lower()

    # 根据文件扩展名选择解析器
    if file_ext == '.pdf':
        return _parse_pdf(file_path)
    elif file_ext in ['.txt', '.md']:
        return _parse_text(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")


def _parse_pdf(file_path: str) -> List[Document]:
    """
    解析 PDF 文件。

    使用 pypdf 进行轻量级解析。对于复杂文档，
    可以切换到 pdfplumber 或其他解析器。

    Args:
        file_path: PDF 文件路径。

    Returns:
        Document 对象列表。
    """
    import pypdf

    docs = []
    source_name = Path(file_path).name

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # 跳过空页面
                if text and text.strip():
                    # 基础清理：移除多余的空白字符
                    text = ' '.join(text.split())

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "filename": source_name,
                            "page": page_num + 1,
                            "page_count": num_pages,
                            "file_type": "pdf",
                        }
                    )
                    docs.append(doc)

        return docs

    except Exception as e:
        # 回退到 pdfplumber（如果可用）
        try:
            import pdfplumber
            return _parse_pdf_with_pdfplumber(file_path)
        except ImportError:
            raise RuntimeError(f"PDF 解析失败: {e}")


def _parse_pdf_with_pdfplumber(file_path: str) -> List[Document]:
    """
    使用 pdfplumber 解析 PDF（备用解析器）。

    Args:
        file_path: PDF 文件路径。

    Returns:
        Document 对象列表。
    """
    import pdfplumber

    docs = []
    source_name = Path(file_path).name

    with pdfplumber.open(file_path) as pdf:
        num_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and text.strip():
                text = ' '.join(text.split())

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "filename": source_name,
                        "page": page_num + 1,
                        "page_count": num_pages,
                        "file_type": "pdf",
                    }
                )
                docs.append(doc)

    return docs


def _parse_text(file_path: str) -> List[Document]:
    """
    解析纯文本文件（.txt, .md）。

    Args:
        file_path: 文本文件路径。

    Returns:
        Document 对象列表。
    """
    import codecs

    docs = []
    source_name = Path(file_path).name

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按段落分割
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    for idx, paragraph in enumerate(paragraphs):
        doc = Document(
            page_content=paragraph,
            metadata={
                "source": file_path,
                "filename": source_name,
                "paragraph": idx + 1,
                "file_type": Path(file_path).suffix.lstrip('.'),
            }
        )
        docs.append(doc)

    return docs


def parse_files(file_paths: List[str]) -> List[Document]:
    """
    批量解析多个文件。

    Args:
        file_paths: 文件路径列表。

    Returns:
        所有文件的 Document 对象列表。
    """
    all_docs = []

    for file_path in file_paths:
        try:
            docs = parse_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"警告: 解析文件 {file_path} 失败: {e}")

    return all_docs