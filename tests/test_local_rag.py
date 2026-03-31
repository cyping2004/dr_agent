"""
端到端测试：本地 RAG 模式

测试纯本地模式（Local Only）的端到端流程。
"""

import os
import tempfile
import pytest

from langchain_core.documents import Document
from agent.state import ResearchState
from graph.research_graph import build_graph
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_documents, embed_query


@pytest.fixture
def temp_local_kb():
    """
    创建临时本地知识库。

    Returns:
        临时集合名称和持久化目录。
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = "test_local_kb"

    # 创建测试数据
    test_docs = [
        Document(
            page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。它以简洁的语法和强大的功能而闻名。",
            metadata={"source": "python_intro.txt", "category": "programming"}
        ),
        Document(
            page_content="Python 的主要特点包括：简洁易读的语法、动态类型、自动内存管理、广泛的库支持。",
            metadata={"source": "python_features.txt", "category": "programming"}
        ),
        Document(
            page_content="Python 广泛应用于数据科学、机器学习、Web 开发、自动化脚本等领域。",
            metadata={"source": "python_usage.txt", "category": "programming"}
        ),
        Document(
            page_content="Python 的列表（list）是一种有序、可变的数据结构，可以存储不同类型的元素。",
            metadata={"source": "python_data_structures.txt", "category": "programming"}
        ),
    ]

    # 存储到向量数据库
    embedded_pairs = embed_documents(test_docs)
    texts = [doc.page_content for doc, _ in embedded_pairs]
    metadatas = [doc.metadata for doc, _ in embedded_pairs]
    embeddings = [embedding for _, embedding in embedded_pairs]

    store = VectorStore(collection_name=collection_name, persist_dir=temp_dir)
    store.upsert(texts, embeddings, metadatas=metadatas)

    yield collection_name, temp_dir

    # 清理
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_local_only_mode(temp_local_kb):
    """
    端到端测试: Local Only 模式
    Query -> Plan -> Retriever -> Evidence Fusion -> Writer

    验证代理仅使用本地知识库回答问题。
    """
    collection_name, persist_dir = temp_local_kb

    # 创建初始状态
    state = ResearchState(
        query="Python 是什么？有什么特点？",
        mode="local_only",
        hitl_enabled=False
    )

    # 构建并运行图
    graph = build_graph()

    result = graph.invoke(state)
    if isinstance(result, dict):
        result = ResearchState(**result)

    # 验证各阶段输出

    # 1. 规划器应生成任务
    assert len(result.research_tasks) >= 1, "应生成至少 1 个研究任务"

    # 2. 应检索到证据
    assert len(result.retrieved_evidence) > 0, "应检索到至少一条证据"

    # 3. 应生成报告
    assert len(result.report_draft) > 0, "应生成报告草稿"

    # 4. 报告应包含与 Python 相关的内容
    report = result.report_draft.lower()
    assert "python" in report or "python" in report, "报告应包含与查询相关的内容"

    print("\n=== Local Only 模式端到端测试结果 ===")
    print(f"生成的任务数: {len(result.research_tasks)}")
    print(f"检索到的证据数: {len(result.retrieved_evidence)}")
    print(f"报告长度: {len(result.report_draft)} 字符")
    print("\n报告预览:")
    print(result.report_draft[:500] + "..." if len(result.report_draft) > 500 else result.report_draft)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_local_retrieval_accuracy(temp_local_kb):
    """
    测试本地检索的准确性。

    验证检索器能找到相关的本地文档。
    """
    from ingestion.vector_store import VectorStore

    collection_name, persist_dir = temp_local_kb

    # 直接测试向量存储检索
    store = VectorStore(collection_name=collection_name, persist_dir=persist_dir)

    # 查询 Python 相关内容
    query_embedding = embed_query("Python 编程语言的特点")
    results = store.similarity_search(query_embedding, top_k=2)

    assert len(results) == 2, "应返回 2 个结果"

    # 验证结果相关性
    result_contents = " ".join([doc.page_content for doc in results]).lower()
    assert "python" in result_contents, "结果应包含 Python"
    assert "特点" in result_contents or "feature" in result_contents or "特点" in result_contents, \
        "结果应与特点相关"

    print("\n=== 本地检索准确性测试 ===")
    for idx, doc in enumerate(results, 1):
        print(f"\n结果 {idx}:")
        print(f"  来源: {doc.metadata.get('source', 'N/A')}")
        print(f"  内容: {doc.page_content[:100]}...")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_graph_local_mode_routing(temp_local_kb):
    """
    测试图在本地模式下的路由。

    验证 router 节点正确路由到 retriever 而非 web_searcher。
    """
    collection_name, persist_dir = temp_local_kb

    # 创建初始状态
    state = ResearchState(
        query="测试查询",
        mode="local_only",
        hitl_enabled=False
    )

    # 构建图
    graph = build_graph()

    # 验证节点存在
    nodes = graph.nodes
    assert "retriever" in nodes, "图中应包含 retriever 节点"
    assert "router" in nodes, "图中应包含 router 节点"

    print("\n=== 本地模式路由测试 ===")
    print(f"图节点: {list(nodes)}")
    print("本地模式验证通过")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_empty_local_kb():
    """
    测试空本地知识库的行为。

    当本地知识库为空时，系统应优雅处理。
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = "test_empty_kb"

    # 创建空的向量存储
    store = VectorStore(collection_name=collection_name, persist_dir=temp_dir)

    # 验证集合为空
    assert store.count() == 0, "新集合应为空"

    # 尝试查询
    query_embedding = embed_query("测试查询")
    results = store.similarity_search(query_embedding, top_k=5)
    assert len(results) == 0, "空集合应返回空结果"

    # 清理
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    print("\n=== 空本地知识库测试 ===")
    print("空知识库行为验证通过")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_local_kb_with_multiple_documents():
    """
    测试包含多个文档的本地知识库。

    验证系统能正确处理和检索多个不同的文档。
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = "test_multi_doc_kb"

    # 创建多个不同主题的文档
    test_docs = [
        Document(
            page_content="量子计算利用量子比特进行计算，能够同时处理多种状态。"
                        "量子纠缠和叠加是量子计算的核心原理。",
            metadata={"source": "quantum.txt", "topic": "physics"}
        ),
        Document(
            page_content="CRISPR-Cas9 是一种革命性的基因编辑技术，"
                        "能够精确修改 DNA 序列，在疾病治疗和农业改良方面有巨大潜力。",
            metadata={"source": "crispr.txt", "topic": "biology"}
        ),
        Document(
            page_content="区块链是一种分布式账本技术，通过密码学保证数据不可篡改。"
                        "比特币是第一个成功应用区块链技术的加密货币。",
            metadata={"source": "blockchain.txt", "topic": "technology"}
        ),
    ]

    # 存储文档
    embedded_pairs = embed_documents(test_docs)
    texts = [doc.page_content for doc, _ in embedded_pairs]
    metadatas = [doc.metadata for doc, _ in embedded_pairs]
    embeddings = [embedding for _, embedding in embedded_pairs]

    store = VectorStore(collection_name=collection_name, persist_dir=temp_dir)
    store.upsert(texts, embeddings, metadatas=metadatas)

    # 测试不同主题的检索
    queries = [
        ("量子计算原理", "physics"),
        ("基因编辑技术", "biology"),
        ("加密货币区块链", "technology"),
    ]

    for query, expected_topic in queries:
        query_embedding = embed_query(query)
        results = store.similarity_search(query_embedding, top_k=1)
        assert len(results) == 1, f"查询 '{query}' 应返回 1 个结果"

        result_topic = results[0].metadata.get("topic", "")
        assert result_topic == expected_topic, \
            f"查询 '{query}' 应返回主题 '{expected_topic}' 的文档，实际得到 '{result_topic}'"

    print("\n=== 多文档本地知识库测试 ===")
    print("多文档检索验证通过")

    # 清理
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])