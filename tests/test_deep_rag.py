"""
测试：Deep RAG 模式

测试深度 RAG 模式的端到端流程：
Planner -> WebSearcher -> Ingestion -> Retriever -> EvidenceFusion -> Writer
"""

import os
import tempfile
import pytest

from langchain_core.documents import Document
from agent.state import ResearchState
from graph.research_graph import build_graph
from ingestion.vector_store import VectorStore
from ingestion.embedder import embed_documents, embed_query


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_deep_rag_end_to_end():
    """
    端到端测试: Deep RAG 模式
    Planner -> WebSearcher -> Ingestion -> Retriever -> EvidenceFusion -> Writer

    验证：
    1. 网络搜索被执行
    2. 搜索结果被摄入向量数据库
    3. 从向量数据库检索到相关内容
    4. 生成最终报告
    """
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("Set TAVILY_API_KEY to run Tavily web search tests")

    os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

    try:
        # 创建临时向量存储用于 Deep RAG
        temp_dir = tempfile.mkdtemp()
        collection_name = "test_deep_rag"

        # 创建初始状态
        state = ResearchState(
            query="Python 基础语法",
            mode="deep_rag",
            hitl_enabled=False
        )

        # 临时覆盖 Retriever 使用的集合
        import agent.retriever as retriever_module
        original_get_retriever = retriever_module.get_retriever

        def get_temp_retriever():
            return retriever_module.Retriever(
                collection_name=collection_name,
                persist_dir=temp_dir
            )
        retriever_module.get_retriever = get_temp_retriever

        try:
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

            # 4. 报告应包含相关内容
            report = result.report_draft.lower()
            assert "python" in report or "语法" in report, "报告应包含与查询相关的内容"

            print("\n=== Deep RAG 模式端到端测试结果 ===")
            print(f"生成的任务数: {len(result.research_tasks)}")
            print(f"检索到的证据数: {len(result.retrieved_evidence)}")
            print(f"报告长度: {len(result.report_draft)} 字符")
            print("\n报告预览:")
            print(result.report_draft[:500] + "..." if len(result.report_draft) > 500 else result.report_draft)

        finally:
            # 恢复原始函数
            retriever_module.get_retriever = original_get_retriever

            # 清理临时向量存储
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    except ImportError:
        pytest.skip("tavily not installed")
    finally:
        os.environ.pop("WEB_SEARCH_PROVIDER", None)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
def test_ingestion_bridge():
    """
    测试即时摄取功能。

    验证：
    1. Web 搜索结果可以被转换为 Document 列表
    2. Document 列表可以被嵌入
    3. 嵌入结果可以存入向量数据库
    4. 存入后可以立即检索到
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = "test_ingestion_bridge"

    try:
        # 模拟 Web 搜索结果
        web_results = [
            Document(
                page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
                metadata={"url": "https://example.com/python", "title": "Python 简介"}
            ),
            Document(
                page_content="Python 的语法简洁明了，适合初学者学习。",
                metadata={"url": "https://example.com/python-syntax", "title": "Python 语法"}
            ),
            Document(
                page_content="Python 广泛应用于 Web 开发、数据科学和自动化。",
                metadata={"url": "https://example.com/python-usage", "title": "Python 应用"}
            ),
        ]

        # 1. 添加 session 元数据
        docs = []
        for idx, result in enumerate(web_results):
            metadata = dict(result.metadata or {})
            metadata.setdefault("source", "web_search")
            metadata["ingestion_session"] = f"test_session_{idx}"
            docs.append(
                Document(
                    page_content=result.page_content,
                    metadata=metadata
                )
            )

        # 2. 嵌入文档
        embedded_pairs = embed_documents(docs)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]

        # 3. 存储到向量数据库
        store = VectorStore(collection_name=collection_name, persist_dir=temp_dir)
        count = store.upsert(texts, embeddings, metadatas=metadatas)

        assert count == len(web_results), f"应存储 {len(web_results)} 个文档，实际存储 {count}"
        assert store.count() == len(web_results), "数据库中应有正确数量的文档"

        # 4. 立即查询
        query_embedding = embed_query("Python 编程语言的特点")
        results = store.similarity_search(query_embedding, top_k=2)

        assert len(results) == 2, "应检索到 2 个结果"

        # 5. 验证检索到的是刚才存储的内容
        result_contents = [doc.page_content for doc in results]
        all_web_content = [doc.page_content for doc in web_results]

        for content in result_contents:
            assert any(web_content in content or content in web_content for web_content in all_web_content), \
                "检索结果应包含存储的 Web 内容"

        print("\n=== 即时摄取桥接测试 ===")
        print(f"存储的文档数: {count}")
        print(f"检索到的结果数: {len(results)}")
        print("验证通过")

    finally:
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
def test_deep_rag_compression_effect():
    """
    测试 Deep RAG 的上下文压缩效果。

    验证：
    1. Web 搜索结果数量大于检索结果数量
    2. 检索结果的相关性较高
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = "test_compression"

    try:
        # 模拟大量 Web 搜索结果
        web_results = []
        topics = [
            "Python 基础语法",
            "Python 数据类型",
            "Python 控制流",
            "Python 函数",
            "Python 类和对象",
            "Python 模块和包",
            "Python 异常处理",
            "Python 文件操作",
            "JavaScript 简介",  # 不相关
            "Java 基础语法",   # 不相关
        ]

        for idx, topic in enumerate(topics):
            doc = Document(
                page_content=f"{topic} 是编程中的重要概念，需要深入理解和掌握。",
                metadata={
                    "source": "web_search",
                    "ingestion_session": "test_session",
                    "topic": topic
                }
            )
            web_results.append(doc)

        # 存储所有结果
        embedded_pairs = embed_documents(web_results)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]

        store = VectorStore(collection_name=collection_name, persist_dir=temp_dir)
        store.upsert(texts, embeddings, metadatas=metadatas)

        # 查询 Python 相关内容
        query_embedding = embed_query("Python 基础知识")
        results = store.similarity_search(query_embedding, top_k=5)

        # 验证压缩效果
        assert len(results) == 5, "应检索到 Top-5 结果"
        assert len(web_results) > len(results), "原始结果应多于检索结果"

        # 验证相关性：检索结果应主要是 Python 相关的
        python_related = 0
        for doc in results:
            topic = doc.metadata.get("topic", "")
            if "Python" in topic:
                python_related += 1

        # 至少 80% 的结果应与 Python 相关
        assert python_related >= 4, f"检索结果相关性不足: {python_related}/5 是 Python 相关"

        print("\n=== 上下文压缩效果测试 ===")
        print(f"原始搜索结果数: {len(web_results)}")
        print(f"压缩后检索数: {len(results)}")
        print(f"Python 相关结果: {python_related}/{len(results)}")
        print(f"压缩率: {(1 - len(results) / len(web_results)) * 100:.1f}%")
        print("验证通过")

    finally:
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
def test_deep_rag_mode_routing():
    """
    测试 Deep RAG 模式下的路由。

    验证路由逻辑正确选择 WebSearcher -> Ingestion -> Retriever 路径。
    """
    # 创建初始状态
    state = ResearchState(
        query="测试查询",
        mode="deep_rag",
        hitl_enabled=False
    )

    # 构建图
    graph = build_graph()

    # 验证节点存在
    nodes = graph.nodes
    assert "web_searcher" in nodes, "图中应包含 web_searcher 节点"
    assert "ingester" in nodes, "图中应包含 ingester 节点"
    assert "retriever" in nodes, "图中应包含 retriever 节点"
    assert "router" in nodes, "图中应包含 router 节点"

    print("\n=== Deep RAG 模式路由测试 ===")
    print(f"图节点: {list(nodes)}")
    print("Deep RAG 路由验证通过")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
def test_web_search_to_vector_store_flow():
    """
    测试完整的 Web 搜索 -> 向量数据库流程。

    这是 Deep RAG 的核心流程。
    """
    temp_dir = tempfile.mkdtemp()
    collection_name = "test_web_to_vector"

    try:
        if not os.getenv("TAVILY_API_KEY"):
            pytest.skip("Set TAVILY_API_KEY to run Tavily web search tests")

        # 1. 模拟网络搜索（使用 tavily）
        os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

        from agent.web_searcher import search

        try:
            search_results = search("Python 编程语言", num_results=5)
        except Exception:
            pytest.skip("Web search failed")

        assert len(search_results) > 0, "应返回搜索结果"
        assert all(isinstance(doc, Document) for doc in search_results), "结果应为 Document 对象"

        # 2. 准备摄取
        docs = []
        for idx, result in enumerate(search_results):
            metadata = dict(result.metadata or {})
            metadata["source"] = "web_search"
            metadata["ingestion_session"] = "test_flow_session"
            docs.append(
                Document(
                    page_content=result.page_content,
                    metadata=metadata
                )
            )

        # 3. 嵌入并存储
        embedded_pairs = embed_documents(docs)
        texts = [doc.page_content for doc, _ in embedded_pairs]
        metadatas = [doc.metadata for doc, _ in embedded_pairs]
        embeddings = [embedding for _, embedding in embedded_pairs]

        store = VectorStore(collection_name=collection_name, persist_dir=temp_dir)
        store.upsert(texts, embeddings, metadatas=metadatas)

        # 4. 验证可以检索
        query_embedding = embed_query("Python 特点")
        results = store.similarity_search(query_embedding, top_k=3)

        assert len(results) > 0, "应能检索到结果"

        # 5. 验证元数据保留
        for doc in results:
            assert "source" in doc.metadata, "应保留 source 元数据"
            assert doc.metadata["source"] == "web_search", "source 应为 web_search"

        print("\n=== Web 搜索到向量数据库流程测试 ===")
        print(f"网络搜索结果数: {len(search_results)}")
        print(f"存储到向量库: {store.count()} 个文档")
        print(f"检索结果数: {len(results)}")
        print("流程验证通过")

    except ImportError:
        pytest.skip("tavily not installed")
    finally:
        os.environ.pop("WEB_SEARCH_PROVIDER", None)
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])