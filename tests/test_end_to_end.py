"""
端到端测试：极速网络模式
"""

import os
import pytest

from agent.state import ResearchState
from graph.research_graph import build_graph


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_fast_web_end_to_end():
    """
    端到端测试: Fast Web 模式
    Query -> Plan -> Web Search -> Evidence Fusion -> Writer
    """
    # 设置为 duckduckgo 以避免需要 TAVILY_API_KEY
    os.environ["WEB_SEARCH_PROVIDER"] = "duckduckgo"

    try:
        # 创建初始状态
        state = ResearchState(
            query="Python 基础语法",
            mode="fast_web",
            auto_approve=True
        )

        # 构建并运行图
        graph = build_graph()

        result = graph.invoke(state)
        if isinstance(result, dict):
            result = ResearchState(**result)

        # 验证各阶段输出

        # 1. 规划器应生成任务
        assert len(result.research_tasks) >= 3, "应生成至少 3 个研究任务"
        assert all(task.strip() for task in result.research_tasks), "所有任务不应为空"

        # 2. 应检索到证据
        assert len(result.retrieved_evidence) > 0, "应检索到至少一条证据"

        # 3. 应生成报告
        assert len(result.report_draft) > 0, "应生成报告草稿"

        # 4. 报告应包含关键部分
        report = result.report_draft.lower()
        assert any(keyword in report for keyword in ["执行摘要", "关键发现", "结论"]), \
            "报告应包含主要章节"

        print("\n=== 端到端测试结果 ===")
        print(f"生成的任务数: {len(result.research_tasks)}")
        print(f"检索到的证据数: {len(result.retrieved_evidence)}")
        print(f"报告长度: {len(result.report_draft)} 字符")
        print("\n报告预览:")
        print(result.report_draft[:500] + "..." if len(result.report_draft) > 500 else result.report_draft)

    except ImportError:
        pytest.skip("duckduckgo-search not installed")
    finally:
        # 恢复原始设置
        os.environ.pop("WEB_SEARCH_PROVIDER", None)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_graph_structure():
    """测试图结构的正确性"""
    graph = build_graph()

    # 验证图已编译
    assert graph is not None, "图应成功编译"

    # 验证节点存在
    nodes = graph.nodes
    expected_nodes = ["planner", "web_searcher", "evidence_fusion", "writer"]
    for node in expected_nodes:
        assert node in nodes, f"图中应包含节点: {node}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])