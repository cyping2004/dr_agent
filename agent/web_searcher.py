"""
网络搜索器
从网络获取实时信息。
"""

import os
import re
from typing import List

from langchain_core.documents import Document
from tavily import TavilyClient
from dotenv import load_dotenv

_DDGS_TEXT_BACKENDS = ("duckduckgo", "bing", "brave", "mojeek", "yahoo")

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
        try:
            return _search_tavily(query, num_results)
        except Exception:
            return _search_ddgs(query, num_results)
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
    使用 DuckDuckGo 兼容后端进行搜索（无需 API 密钥）。

    Args:
        query: 搜索查询。
        num_results: 返回结果数量。

    Returns:
        List of Document objects.
    """
    try:
        return _search_ddgs(query, num_results)
    except Exception:
        return _search_tavily(query, num_results)


def _search_ddgs(query: str, num_results: int) -> List[Document]:
    """
    使用 DDGS 的多个文本搜索后端进行搜索。

    该实现优先尝试 DuckDuckGo；如果当前网络环境无法访问，再按顺序
    回退到 Bing、Brave、Mojeek 等可用后端，避免默认后端失效导致整条
    搜索链路中断。

    Args:
        query: 搜索查询。
        num_results: 返回结果数量。

    Returns:
        List of Document objects。
    """
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError:
        raise ImportError(
            "ddgs not installed. "
            "Install it with: pip install ddgs"
        )

    last_error = None

    for backend in _DDGS_TEXT_BACKENDS:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results, backend=backend))

                docs = []
                for result in results:
                    url = result.get("href") or result.get("url") or ""
                    title = result.get("title", "")
                    original_content = result.get("body", "") or result.get("snippet", "")
                    content = original_content

                    if url:
                        try:
                            extracted = ddgs.extract(url, fmt="text_plain")
                            extracted_content = extracted.get("content", "") or ""
                            if extracted_content and not extracted_content.lstrip().startswith("%PDF-"):
                                content = extracted_content
                        except Exception:
                            pass

                    if content.lstrip().startswith("%PDF-"):
                        content = original_content or title or url

                    content = re.sub(r"<[^>]+>", "", content)
                    content = re.sub(r"\s+", " ", content).strip()

                    if len(content) > 4000:
                        content = content[:4000] + "..."

                    if not content:
                        content = title or url

                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "url": url,
                                "title": title,
                                "source": "web",
                                "search_backend": backend,
                            },
                        )
                    )

                if docs:
                    return docs
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise DDGSException(str(last_error))

    raise DDGSException("No results found.")