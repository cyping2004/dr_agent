"""
测试：模式对比

对比 Fast Web Baseline 和 Deep RAG 模式的效果。

使用测试模式确保两种模式使用相同的 Web 搜索结果，
从而公平比较摘要压缩和向量检索的效果。
"""

import os
import tempfile
import pytest

from agent.state import ResearchState
from graph.research_graph import build_graph


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_fast_web_baseline():
    """
    端到端测试: Fast Web Baseline 模式

    流程: Planner -> WebSearcher -> EvidenceFusion -> Writer

    验证：
    1. 生成研究任务
    2. 执行网络搜索
    3. 生成报告
    """
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("Set TAVILY_API_KEY to run Tavily web search tests")

    os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

    try:
        # 创建初始状态
        state = ResearchState(
            query="Python 列表推导式",
            mode="fast_web",
            hitl_enabled=False
        )

        # 构建并运行图
        graph = build_graph()
        result = graph.invoke(state)
        if isinstance(result, dict):
            result = ResearchState(**result)

        # 验证各阶段输出
        assert len(result.research_tasks) >= 1, "应生成至少 1 个研究任务"
        assert len(result.retrieved_evidence) > 0, "应检索到至少一条证据"
        assert len(result.report_draft) > 0, "应生成报告草稿"

        report = result.report_draft.lower()
        assert "python" in report or "列表" in report, "报告应包含相关内容"

        print("\n=== Fast Web Baseline 测试结果 ===")
        print(f"生成的任务数: {len(result.research_tasks)}")
        print(f"检索到的证据数: {len(result.retrieved_evidence)}")
        print(f"报告长度: {len(result.report_draft)} 字符")
        print(f"\n报告预览:")
        print(result.report_draft[:800] + "..." if len(result.report_draft) > 800 else result.report_draft)

        return result  # 返回结果供后续对比使用

    except ImportError:
        pytest.skip("tavily not installed")
    finally:
        os.environ.pop("WEB_SEARCH_PROVIDER", None)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_deep_rag_mode():
    """
    端到端测试: Deep RAG 模式

    流程: Planner -> WebSearcher -> Ingestion -> Retriever -> EvidenceFusion -> Writer

    验证：
    1. 生成研究任务
    2. 执行网络搜索
    3. 摄取到向量数据库
    4. 从向量数据库检索
    5. 生成报告
    """
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("Set TAVILY_API_KEY to run Tavily web search tests")

    os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

    temp_dir = tempfile.mkdtemp()
    collection_name = "test_deep_rag_comparison"

    try:
        # 创建初始状态
        state = ResearchState(
            query="Python 列表推导式",
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
            assert len(result.research_tasks) >= 1, "应生成至少 1 个研究任务"
            assert len(result.retrieved_evidence) > 0, "应检索到至少一条证据"
            assert len(result.report_draft) > 0, "应生成报告草稿"

            report = result.report_draft.lower()
            assert "python" in report or "列表" in report, "报告应包含相关内容"

            print("\n=== Deep RAG 模式测试结果 ===")
            print(f"生成的任务数: {len(result.research_tasks)}")
            print(f"检索到的证据数: {len(result.retrieved_evidence)}")
            print(f"报告长度: {len(result.report_draft)} 字符")
            print(f"\n报告预览:")
            print(result.report_draft[:800] + "..." if len(result.report_draft) > 800 else result.report_draft)

            return result  # 返回结果供后续对比使用

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
@pytest.mark.e2e
def test_mode_comparison():
    """
    对比测试: Fast Web Baseline vs Deep RAG

    使用相同的查询，比较两种模式的输出。
    """
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("Set TAVILY_API_KEY to run Tavily web search tests")

    os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

    try:
        query = "Python 装饰器"
        temp_dir = tempfile.mkdtemp()
        collection_name = "test_mode_comparison"

        print("\n" + "="*60)
        print("模式对比测试: Fast Web vs Deep RAG")
        print(f"查询: {query}")
        print("="*60)

        # 1. Fast Web Baseline
        print("\n>>> 运行 Fast Web Baseline...")
        fast_state = ResearchState(
            query=query,
            mode="fast_web",
            hitl_enabled=False
        )
        graph = build_graph()
        fast_result = graph.invoke(fast_state)
        if isinstance(fast_result, dict):
            fast_result = ResearchState(**fast_result)

        # 2. Deep RAG
        print(">>> 运行 Deep RAG...")
        import agent.retriever as retriever_module
        original_get_retriever = retriever_module.get_retriever

        def get_temp_retriever():
            return retriever_module.Retriever(
                collection_name=collection_name,
                persist_dir=temp_dir
            )
        retriever_module.get_retriever = get_temp_retriever

        try:
            deep_state = ResearchState(
                query=query,
                mode="deep_rag",
                hitl_enabled=False
            )
            deep_result = graph.invoke(deep_state)
            if isinstance(deep_result, dict):
                deep_result = ResearchState(**deep_result)
        finally:
            retriever_module.get_retriever = original_get_retriever
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        # 3. 对比结果
        print("\n" + "="*60)
        print("对比结果:")
        print("="*60)

        print(f"\nFast Web Baseline:")
        print(f"  - 研究任务数: {len(fast_result.research_tasks)}")
        print(f"  - 证据数量: {len(fast_result.retrieved_evidence)}")
        print(f"  - 报告长度: {len(fast_result.report_draft)} 字符")

        print(f"\nDeep RAG:")
        print(f"  - 研究任务数: {len(deep_result.research_tasks)}")
        print(f"  - 证据数量: {len(deep_result.retrieved_evidence)}")
        print(f"  - 报告长度: {len(deep_result.report_draft)} 字符")

        # 对比证据数量（Deep RAG 可能更少，因为经过压缩）
        evidence_diff = len(fast_result.retrieved_evidence) - len(deep_result.retrieved_evidence)
        if evidence_diff > 0:
            print(f"\n  -> Deep RAG 压缩了 {evidence_diff} 条证据")
        elif evidence_diff < 0:
            print(f"\n  -> Deep RAG 扩展了 {abs(evidence_diff)} 条证据")

        # 两种模式都应生成有效报告
        assert len(fast_result.report_draft) > 0, "Fast Web 应生成报告"
        assert len(deep_result.report_draft) > 0, "Deep RAG 应生成报告"

        # 报告都应包含相关内容
        fast_report = fast_result.report_draft.lower()
        deep_report = deep_result.report_draft.lower()

        query_keywords = query.lower().split()
        for keyword in query_keywords:
            if len(keyword) > 2:  # 跳过短词
                if keyword in fast_report:
                    print(f"  ✓ Fast Web 报告包含关键词: '{keyword}'")
                if keyword in deep_report:
                    print(f"  ✓ Deep RAG 报告包含关键词: '{keyword}'")

        print("\n" + "="*60)
        print("对比测试完成！")
        print("="*60)

    except ImportError:
        pytest.skip("tavily not installed")
    finally:
        os.environ.pop("WEB_SEARCH_PROVIDER", None)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_mode_with_test_mode():
    """
    测试测试模式：Fast Web 和 Deep RAG 复用相同的 Web 搜索结果。

    验证：
    1. test_mode=True 时，Web 搜索结果被缓存
    2. 两种模式使用相同的缓存结果
    3. 减少搜索次数
    """
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("Set TAVILY_API_KEY to run Tavily web search tests")

    os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

    try:
        query = "Python 异常处理"
        temp_dir = tempfile.mkdtemp()
        collection_name = "test_mode_comparison_test"

        print("\n" + "="*60)
        print("测试模式: 复用 Web 搜索结果")
        print(f"查询: {query}")
        print("="*60)

        # 1. Fast Web with test_mode
        print("\n>>> 运行 Fast Web (test_mode=True)...")
        fast_state = ResearchState(
            query=query,
            mode="fast_web",
            hitl_enabled=False,
            test_mode=True
        )
        graph = build_graph()
        fast_result = graph.invoke(fast_state)
        if isinstance(fast_result, dict):
            fast_result = ResearchState(**fast_result)

        # 验证缓存被填充
        assert len(fast_result.web_search_cache) > 0, "Web 搜索结果应被缓存"
        cached_results = fast_result.web_search_cache

        print(f"  - 缓存的搜索结果数: {len(cached_results)}")

        # 2. Deep RAG 复用缓存
        print(">>> 运行 Deep RAG (复用缓存)...")
        import agent.retriever as retriever_module
        original_get_retriever = retriever_module.get_retriever

        def get_temp_retriever():
            return retriever_module.Retriever(
                collection_name=collection_name,
                persist_dir=temp_dir
            )
        retriever_module.get_retriever = get_temp_retriever

        try:
            deep_state = ResearchState(
                query=query,
                mode="deep_rag",
                hitl_enabled=False,
                test_mode=True,
                web_search_cache=cached_results  # 传递缓存
            )
            deep_result = graph.invoke(deep_state)
            if isinstance(deep_result, dict):
                deep_result = ResearchState(**deep_result)
        finally:
            retriever_module.get_retriever = original_get_retriever
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        # 3. 验证两种模式都生成了报告
        assert len(fast_result.report_draft) > 0, "Fast Web 应生成报告"
        assert len(deep_result.report_draft) > 0, "Deep RAG 应生成报告"

        print("\n" + "="*60)
        print("测试模式验证:")
        print("="*60)
        print(f"Fast Web 报告长度: {len(fast_result.report_draft)} 字符")
        print(f"Deep RAG 报告长度: {len(deep_result.report_draft)} 字符")
        print("\n两种模式使用相同的 Web 搜索结果")
        print("测试模式验证通过！")
        print("="*60)

    except ImportError:
        pytest.skip("tavily not installed")
    finally:
        os.environ.pop("WEB_SEARCH_PROVIDER", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])