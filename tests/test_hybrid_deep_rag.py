"""
测试：hybrid_deep_rag 模式（本地文档 + Web 结果同集合检索）。
"""

import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

import agent.retriever as retriever_module
import agent.writer as writer_module
import graph.research_graph as research_graph
from agent.state import ResearchState
from ingestion.parser import parse_file





def test_pdf_parse(tmp_path):
    pytest.importorskip("reportlab")

    pdf_path = tmp_path / "sample.pdf"
    from reportlab.pdfgen import canvas

    canvas_obj = canvas.Canvas(str(pdf_path))
    canvas_obj.drawString(72, 720, "PDF test content")
    canvas_obj.showPage()
    canvas_obj.save()

    docs = parse_file(str(pdf_path))
    assert docs, "PDF 应解析出至少一个 Document"
    assert any("PDF test content" in doc.page_content for doc in docs)
    assert docs[0].metadata.get("file_type") == "pdf"


def test_txt_parse(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("段落一\n\n段落二", encoding="utf-8")

    docs = parse_file(str(txt_path))
    assert len(docs) == 2
    assert docs[0].page_content == "段落一"
    assert docs[1].page_content == "段落二"
    assert docs[0].metadata.get("file_type") == "txt"


def test_md_parse(tmp_path):
    md_path = tmp_path / "sample.md"
    md_path.write_text("# 标题\n\n- 列表项", encoding="utf-8")

    docs = parse_file(str(md_path))
    assert len(docs) == 2
    assert docs[0].page_content == "# 标题"
    assert docs[1].page_content == "- 列表项"
    assert docs[0].metadata.get("file_type") == "md"


def test_citation_map():
    evidence = [
        Document(
            page_content="网页证据 A",
            metadata={"title": "Doc A", "url": "https://example.com/a", "source_type": "web"},
        ),
        Document(
            page_content="网页证据 A 重复",
            metadata={"title": "Doc A", "url": "https://example.com/a", "source_type": "web"},
        ),
        Document(
            page_content="本地证据",
            metadata={"source_type": "local", "filename": "notes.txt", "source": "/tmp/notes.txt"},
        ),
    ]

    references, evidence_ref_map = writer_module._collect_references(evidence)
    section = writer_module._build_reference_section(references)

    report = "引用 [1] [2] [3]。"
    remapped = writer_module._remap_citations(report, evidence_ref_map)

    cited_numbers = {int(num) for num in re.findall(r"\[(\d+)\]", remapped)}
    ref_numbers = {int(num) for num in re.findall(r"^\[(\d+)\]", section, flags=re.MULTILINE)}

    assert cited_numbers, "应至少包含一个引用编号"
    assert cited_numbers.issubset(ref_numbers), "引用编号应与参考来源一一对应"

def test_ingests_content(monkeypatch, tmp_path):
    """hybrid_deep_rag 应先入库本地文档，再入库 web 结果并联合检索。"""
    local_file = tmp_path / "local_notes.txt"
    local_file.write_text("Python 装饰器用于在不修改函数的前提下扩展行为。", encoding="utf-8")

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOCAL_RETRIEVAL_TOP_K", "50")

    def fake_plan(state: ResearchState) -> ResearchState:
        state.research_tasks = ["Python 装饰器"]
        return state

    def fake_search(query: str, num_results: int = 10):
        return [
            Document(
                page_content="网页资料：装饰器常用于日志记录和权限控制。",
                metadata={
                    "title": "Decorator Guide",
                    "url": "https://example.com/decorator",
                    "source": "web_search",
                },
            )
        ]

    def fake_write(state: ResearchState) -> ResearchState:
        state.report_draft = "# 测试报告\n\n## 1. 执行摘要\n\n引用 [1] 与 [2]。"
        return state

    def fake_embed_documents(docs):
        return [(doc, [0.1, 0.2, 0.3]) for doc in docs]

    def fake_retrieve(self, query: str, top_k: int = None):
        docs = self.vector_store.get_all_documents()
        if top_k:
            return docs[:top_k]
        return docs

    monkeypatch.setattr(research_graph, "plan", fake_plan)
    monkeypatch.setattr(research_graph, "search", fake_search)
    monkeypatch.setattr(research_graph, "write", fake_write)
    monkeypatch.setattr(research_graph, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(retriever_module.Retriever, "retrieve", fake_retrieve)

    graph = research_graph.build_graph()
    result = graph.invoke(
        ResearchState(
            query="解释 Python 装饰器",
            mode="hybrid_deep_rag",
            local_files=[str(Path(local_file))],
        )
    )
    if isinstance(result, dict):
        result = ResearchState(**result)

    assert result.working_collection.startswith("temp_hybrid_")
    assert len(result.retrieved_evidence) > 0

    source_types = {(doc.metadata or {}).get("source_type") for doc in result.retrieved_evidence}
    assert "local" in source_types
    assert "web" in source_types


def test_keeps_local_evidence(monkeypatch, tmp_path):
    """即使 Retriever 只返回 web 结果，也应保底追加 local 证据。"""
    local_file = tmp_path / "local_notes.txt"
    local_file.write_text("本地资料：Deep Research Agent 强调证据融合。", encoding="utf-8")

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOCAL_RETRIEVAL_TOP_K", "20")

    def fake_plan(state: ResearchState) -> ResearchState:
        state.research_tasks = ["Deep Research Agent"]
        return state

    def fake_search(query: str, num_results: int = 10):
        return [
            Document(
                page_content="网页资料：Deep Research Agent 结合规划与检索。",
                metadata={
                    "title": "Deep Research Agent Intro",
                    "url": "https://example.com/deep-research-agent",
                    "source": "web_search",
                },
            )
        ]

    def fake_write(state: ResearchState) -> ResearchState:
        state.report_draft = "# 测试报告\n\n## 1. 执行摘要\n\n引用 [1]。"
        return state

    def fake_embed_documents(docs):
        return [(doc, [0.2, 0.1, 0.3]) for doc in docs]

    def fake_retrieve_web_only(self, query: str, top_k: int = None):
        docs = self.vector_store.get_all_documents()
        web_docs = [
            doc
            for doc in docs
            if str((doc.metadata or {}).get("source_type", "")).strip().lower() == "web"
        ]
        if top_k:
            return web_docs[:top_k]
        return web_docs

    monkeypatch.setattr(research_graph, "plan", fake_plan)
    monkeypatch.setattr(research_graph, "search", fake_search)
    monkeypatch.setattr(research_graph, "write", fake_write)
    monkeypatch.setattr(research_graph, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(retriever_module.Retriever, "retrieve", fake_retrieve_web_only)

    graph = research_graph.build_graph()
    result = graph.invoke(
        ResearchState(
            query="deep research agent",
            mode="hybrid_deep_rag",
            local_files=[str(Path(local_file))],
        )
    )
    if isinstance(result, dict):
        result = ResearchState(**result)

    source_types = [
        str((doc.metadata or {}).get("source_type", "")).strip().lower()
        for doc in result.retrieved_evidence
    ]
    assert "web" in source_types
    assert "local" in source_types