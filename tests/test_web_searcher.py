"""
测试：网络搜索器
"""

import os
import pytest

from agent.web_searcher import search


@pytest.mark.skipif(
    os.getenv("WEB_SEARCH_PROVIDER", "tavily") == "tavily" and not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set for tavily provider"
)
def test_web_search():
    """测试网络搜索功能"""
    query = "AI Agent trends 2024"

    try:
        results = search(query, num_results=3)
    except Exception as exc:
        pytest.skip(f"web search failed: {exc}")

    assert len(results) > 0, "应该返回搜索结果"
    assert all(hasattr(result, "page_content") for result in results), "每个结果应有 page_content"
    assert all("url" in result.metadata for result in results), "每个结果应有 URL 元数据"

    # 验证 URL 格式
    for result in results:
        url = result.metadata["url"]
        assert url.startswith("http"), f"URL 应以 http 或 https 开头: {url}"


# @pytest.mark.skipif(
#     os.getenv("WEB_SEARCH_PROVIDER", "tavily") == "tavily" and not os.getenv("TAVILY_API_KEY"),
#     reason="TAVILY_API_KEY not set for tavily provider"
# )
def test_web_search_with_duckduckgo():
    """测试 DuckDuckGo 搜索（无需 API 密钥）"""
    # 临时设置环境变量使用 duckduckgo
    original_provider = os.environ.get("WEB_SEARCH_PROVIDER")
    os.environ["WEB_SEARCH_PROVIDER"] = "duckduckgo"

    try:
        query = "Python programming"
        try:
            results = search(query, num_results=2)
        except Exception as exc:
            pytest.skip(f"duckduckgo search failed: {exc}")

        assert len(results) > 0, "DuckDuckGo 应返回搜索结果"
        assert all("url" in result.metadata for result in results), "每个结果应有 URL 元数据"
    except ImportError:
        pytest.skip("duckduckgo-search not installed")
    finally:
        if original_provider:
            os.environ["WEB_SEARCH_PROVIDER"] = original_provider
        else:
            os.environ.pop("WEB_SEARCH_PROVIDER", None)


def test_web_search_missing_api_key():
    """测试缺少 API 密钥时的行为"""
    # 临时移除 API 密钥
    original_tavily_key = os.environ.get("TAVILY_API_KEY")
    os.environ.pop("TAVILY_API_KEY", None)
    os.environ["WEB_SEARCH_PROVIDER"] = "tavily"

    try:
        with pytest.raises(ValueError, match="TAVILY_API_KEY not set"):
            search("test query")
    finally:
        if original_tavily_key:
            os.environ["TAVILY_API_KEY"] = original_tavily_key