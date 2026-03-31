"""
Writer formatting test using a real LLM, with simulated evidence inputs.
"""

import os
import re

import pytest

import agent.writer as writer_module
from agent.state import ResearchState


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run writer LLM test"
)
def test_writer_postprocess_evidence_urls():
    """Ensure evidence list is renumbered and URLs are appended."""
    state = ResearchState(query="测试")
    state.messages.append({
        "role": "evidence_index_map",
        "content": [
            {"index": 1, "url": "https://example.com/1"},
            {"index": 2, "url": "https://example.com/2"},
            {"index": 3, "url": ""},
        ],
    })
    state.messages.append({
        "role": "evidence_summaries",
        "content": [
            {
                "task": "任务 1",
                "summary": "发现 A [1]，发现 B [2]，发现 C [3]。",
            },
        ],
    })

    result = writer_module.write(state)

    assert "## 证据与来源" in result.report_draft
    assert re.search(r"- \[1\] .+ （url：https://example.com/1）", result.report_draft)
    assert re.search(r"- \[2\] .+ （url：https://example.com/2）", result.report_draft)
    assert re.search(r"- \[3\] .+ （url：未知）", result.report_draft)
