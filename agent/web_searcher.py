"""
网络搜索器
从网络获取实时信息。
"""

import os
from typing import List

from langchain_core.documents import Document
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


def search(query: str, num_results: int = 10) -> List[Document]:
    """
    执行网络搜索，返回结果作为 Document 对象。

    Args:
        query: 搜索查询字符串。
        num_results: 返回结果数量。

    Returns:
        List of Document objects with URL metadata.
    """
    provider = os.getenv("WEB_SEARCH_PROVIDER", "tavily")

    if provider == "tavily":
        return _search_tavily(query, num_results)
    else:
        return _search_duckduckgo(query, num_results)


def _search_tavily(query: str, num_results: int) -> List[Document]:
    """
    使用 Tavily API 进行搜索。

    Args:
        query: 搜索查询。
        num_results: 返回结果数量。

    Returns:
        List of Document objects.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set in environment variables")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=num_results,
        include_answer=False,
        include_raw_content="text",
    )

    docs = []
    for result in response.get("results", []):
        raw_content = result.get("raw_content", "")
        content = raw_content or result.get("content", "")
        url = result.get("url", "")
        title = result.get("title", "")

        # 去除 HTML 标签并截断长内容
        import re
        content = re.sub(r'<[^>]+>', '', content)

        doc = Document(
            page_content=content,
            metadata={
                "url": url,
                "title": title,
                "source": "web",
            }
        )
        docs.append(doc)

    return docs


def _search_duckduckgo(query: str, num_results: int) -> List[Document]:
    """
    使用 DuckDuckGo 进行搜索（无需 API 密钥）。

    Args:
        query: 搜索查询。
        num_results: 返回结果数量。

    Returns:
        List of Document objects.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        raise ImportError(
            "ddgs not installed. "
            "Install it with: pip install ddgs"
        )

    docs = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

        for result in results:
            content = result.get("body", "")
            url = result.get("href", "")
            title = result.get("title", "")

            if len(content) > 2000:
                content = content[:2000] + "..."

            doc = Document(
                page_content=content,
                metadata={
                    "url": url,
                    "title": title,
                    "source": "web",
                }
            )
            docs.append(doc)

    return docs