"""
测试：Writer 引用构建应包含本地文档名。
"""

from langchain_core.documents import Document

import agent.writer as writer_module


def test_reference_section_contains_local_filename():
    evidence = [
        Document(
            page_content="本地 PDF 内容片段 A",
            metadata={"source_type": "local", "filename": "thesis.pdf", "source": "/tmp/thesis.pdf"},
        ),
        Document(
            page_content="本地 PDF 内容片段 B",
            metadata={"source_type": "local", "filename": "thesis.pdf", "source": "/tmp/thesis.pdf"},
        ),
        Document(
            page_content="网页内容",
            metadata={"title": "Web Doc", "url": "https://example.com/doc", "source_type": "web"},
        ),
    ]

    references, evidence_ref_map = writer_module._collect_references(evidence)
    section = writer_module._build_reference_section(references)

    assert evidence_ref_map[0] == evidence_ref_map[1], "同一文档应去重为同一引用编号"
    assert "本地文档: thesis.pdf" in section
    assert "https://example.com/doc" in section
