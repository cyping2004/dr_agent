"""
测试：文档解析器
"""

import os
import tempfile
import pytest

from ingestion.parser import parse_file, parse_files
from langchain_core.documents import Document


class TestParser:
    """测试解析器功能"""

    def test_parse_text_file(self):
        """测试解析文本文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("第一段内容。\n\n第二段内容。\n\n第三段内容。")
            temp_path = f.name

        try:
            docs = parse_file(temp_path)

            assert len(docs) == 3, "应该解析出 3 个段落"
            assert all(isinstance(doc, Document) for doc in docs), "每个结果应为 Document 对象"
            assert all(doc.page_content for doc in docs), "每个文档应有内容"

            # 检查元数据
            assert all("source" in doc.metadata for doc in docs), "应有 source 元数据"
            assert all("file_type" in doc.metadata for doc in docs), "应有 file_type 元数据"

        finally:
            os.unlink(temp_path)

    def test_parse_markdown_file(self):
        """测试解析 Markdown 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# 标题\n\n内容段落 1。\n\n内容段落 2。")
            temp_path = f.name

        try:
            docs = parse_file(temp_path)

            assert len(docs) >= 1, "应至少解析出 1 个段落"
            assert isinstance(docs[0], Document)

        finally:
            os.unlink(temp_path)

    def test_parse_nonexistent_file(self):
        """测试解析不存在的文件"""
        with pytest.raises(FileNotFoundError):
            parse_file("/nonexistent/path/file.txt")

    def test_parse_unsupported_file_type(self):
        """测试不支持的文件类型"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持的文件类型"):
                parse_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_parse_files_batch(self):
        """测试批量解析多个文件"""
        # 创建多个临时文件
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(f"文件 {i} 的内容。\n\n第二个段落。")
                temp_files.append(f.name)

        try:
            docs = parse_files(temp_files)

            # 3 个文件，每个 2 个段落 = 6 个文档
            assert len(docs) == 6, f"应解析出 6 个文档，实际得到 {len(docs)}"

        finally:
            for temp_path in temp_files:
                os.unlink(temp_path)

    def test_parse_empty_text_file(self):
        """测试解析空文本文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")
            temp_path = f.name

        try:
            docs = parse_file(temp_path)
            assert len(docs) == 0, "空文件应返回空列表"

        finally:
            os.unlink(temp_path)

    def test_parse_text_with_extra_whitespace(self):
        """测试解析带有多余空白的文本"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("  多余  空白   的  文本  。\n\n\n\n  另一段  。  ")
            temp_path = f.name

        try:
            docs = parse_file(temp_path)

            assert len(docs) == 2, "应解析出 2 个段落"
            # 检查空白是否被清理
            assert "  " not in docs[0].page_content, "多余空白应被清理"

        finally:
            os.unlink(temp_path)


class TestPDFParser:
    """测试 PDF 解析器"""

    def test_parse_pdf(self, sample_pdf):
        """测试解析 PDF 文件"""
        if sample_pdf is None:
            pytest.skip("未提供测试 PDF 文件。请将 PDF 文件放在 tests/data/sample.pdf")

        docs = parse_file(sample_pdf)

        assert len(docs) > 0, "应解析出至少 1 页"
        assert all(isinstance(doc, Document) for doc in docs), "每个结果应为 Document 对象"

        # 检查 PDF 特定元数据
        assert all("page" in doc.metadata for doc in docs), "应有 page 元数据"
        assert all("page_count" in doc.metadata for doc in docs), "应有 page_count 元数据"
        assert all("file_type" in doc.metadata and doc.metadata["file_type"] == "pdf" for doc in docs), "file_type 应为 pdf"

    def test_parse_pdf_content_quality(self, sample_pdf):
        """测试 PDF 解析内容质量"""
        if sample_pdf is None:
            pytest.skip("未提供测试 PDF 文件。请将 PDF 文件放在 tests/data/sample.pdf")

        docs = parse_file(sample_pdf)

        # 检查内容不为空
        assert all(doc.page_content for doc in docs), "每个文档应有内容"

        # 检查没有乱码（简单检查：没有大量连续的空字符）
        for doc in docs:
            # 计算非空白字符的比例
            text = doc.page_content
            non_space_chars = sum(1 for c in text if not c.isspace())
            total_chars = len(text)

            # 至少 50% 是非空白字符
            if total_chars > 0:
                assert non_space_chars / total_chars > 0.5, f"页面内容可能包含乱码: {text[:50]}..."

    def test_parse_pdf_page_numbers(self, sample_pdf):
        """测试 PDF 页码正确性"""
        if sample_pdf is None:
            pytest.skip("未提供测试 PDF 文件。请将 PDF 文件放在 tests/data/sample.pdf")

        docs = parse_file(sample_pdf)

        # 检查页码连续
        page_numbers = [doc.metadata["page"] for doc in docs]
        expected_pages = list(range(1, len(docs) + 1))

        assert page_numbers == expected_pages, f"页码不连续: {page_numbers}"


# Pytest fixture for sample PDF
@pytest.fixture
def sample_pdf():
    """
    提供测试用的 PDF 文件路径。

    请将测试 PDF 文件放在 tests/data/sample.pdf

    Returns:
        PDF 文件路径，如果文件不存在则返回 None。
    """
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample.pdf")

    if os.path.exists(sample_path):
        return sample_path

    return None