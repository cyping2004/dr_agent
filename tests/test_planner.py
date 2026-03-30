"""
测试：规划器
"""

import os
import pytest

from agent.state import ResearchState
from agent.planner import plan, _parse_planning_response


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("RUN_LLM_TESTS") == "1"),
    reason="Set RUN_LLM_TESTS=1 and OPENAI_API_KEY to run LLM tests"
)
def test_plan_generation():
    """测试规划器生成研究任务"""
    state = ResearchState(query="Quantum Computing")

    result_state = plan(state)

    assert len(result_state.research_tasks) >= 3, "应生成至少 3 个子任务"
    assert len(result_state.research_tasks) <= 7, "应生成不超过 7 个子任务"
    assert all(isinstance(task, str) for task in result_state.research_tasks), "每个任务应为字符串"
    assert all(task.strip() for task in result_state.research_tasks), "每个任务不应为空"


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("RUN_LLM_TESTS") == "1"),
    reason="Set RUN_LLM_TESTS=1 and OPENAI_API_KEY to run LLM tests"
)
def test_plan_with_feedback():
    """测试带反馈的规划"""
    state = ResearchState(
        query="Quantum Computing",
        messages=[{
            "role": "user",
            "content": "Feedback: 请聚焦于量子计算的实际应用"
        }]
    )

    result_state = plan(state)

    assert len(result_state.research_tasks) > 0, "应生成研究任务"

    # 验证任务是否聚焦于应用
    combined_tasks = " ".join(result_state.research_tasks).lower()
    assert any(keyword in combined_tasks for keyword in ["应用", "application", "use case"]), \
        "任务应聚焦于实际应用"


def test_parse_planning_response():
    """测试规划响应解析"""
    # 测试编号列表格式
    response = """1. What is quantum computing?
2. How does it work?
3. What are the applications?"""

    tasks = _parse_planning_response(response)

    assert len(tasks) == 3, "应解析出 3 个任务"
    assert tasks[0] == "What is quantum computing?", "第一个任务应正确解析"
    assert tasks[1] == "How does it work?", "第二个任务应正确解析"
    assert tasks[2] == "What are the applications?", "第三个任务应正确解析"


def test_parse_planning_response_with_bullets():
    """测试带项目符号的响应解析"""
    response = """- First task
- Second task
- Third task"""

    tasks = _parse_planning_response(response)

    assert len(tasks) == 3, "应解析出 3 个任务"


def test_parse_planning_response_plain_text():
    """测试纯文本响应解析"""
    response = "A single task description"

    tasks = _parse_planning_response(response)

    assert len(tasks) == 1, "应解析出 1 个任务"
    assert tasks[0] == "A single task description", "任务应正确解析"


def test_parse_planning_response_empty():
    """测试空响应"""
    tasks = _parse_planning_response("")

    assert len(tasks) == 0, "空响应应返回空列表"