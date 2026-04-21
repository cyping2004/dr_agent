"""
测试：hybrid_deep_rag 模式（本地文档 + Web 结果同集合检索）。
"""

from pathlib import Path

from langchain_core.documents import Document

import agent.retriever as retriever_module
import graph.research_graph as research_graph
from agent.state import ResearchState


def test_hybrid_deep_rag_ingests_local_and_web(monkeypatch, tmp_path):
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


def test_hybrid_deep_rag_force_keeps_local_evidence(monkeypatch, tmp_path):
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
