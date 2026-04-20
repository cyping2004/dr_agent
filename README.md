# 深度研究代理（dr_agent）

本文仅记录当前**已实现**的功能与测试，聚焦实际运行路径与测试覆盖面。

## 已实现功能

### 运行模式

- Fast Web：Planner → WebSearcher → EvidenceFusion → Writer
- Deep RAG：WebSearcher → Ingestion → Retriever → EvidenceFusion → Writer（上下文压缩）
- Local Only：Retriever 直接使用本地向量库（需先用 [ingest_local.py](ingest_local.py) 入库，CLI 暂未暴露该模式）
- Test Mode：复用 `web_search_cache`，保证 fast_web 与 deep_rag 使用同一份搜索结果

### 离线入库

- [ingest_local.py](ingest_local.py) 支持 PDF/TXT/MD
- 解析 → 分块 → 嵌入 → Chroma 持久化

### 核心模块

- 统一状态：ResearchState（[agent/state.py](agent/state.py)）
- 规划器：生成 3-7 个可搜索子任务（[agent/planner.py](agent/planner.py)）
- 网络搜索：Tavily 或 DuckDuckGo（[agent/web_searcher.py](agent/web_searcher.py)）
- 向量检索：按子任务 Top-K 检索（[agent/retriever.py](agent/retriever.py)）
- 证据融合：按任务聚合摘要（[agent/evidence_fusion.py](agent/evidence_fusion.py)）
- 报告生成：Markdown 输出（[agent/writer.py](agent/writer.py)）

### 最低环境要求

- `OPENAI_API_KEY`（LLM 与 embeddings）
- `WEB_SEARCH_PROVIDER=tavily|duckduckgo`
- `TAVILY_API_KEY`（仅 tavily 需要）
- `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION`（可选）

## 测试概览

### 解析与向量库（离线可跑）

- [tests/test_parser.py](tests/test_parser.py)
- [tests/test_vector_db.py](tests/test_vector_db.py)

### Web 搜索

- [tests/test_web_searcher.py](tests/test_web_searcher.py)

### 端到端/模式对比（需 LLM 与 embeddings，部分需要 Tavily）

- [tests/test_end_to_end.py](tests/test_end_to_end.py)
- [tests/test_deep_rag.py](tests/test_deep_rag.py)
- [tests/test_local_rag.py](tests/test_local_rag.py)
- [tests/test_mode_comparison.py](tests/test_mode_comparison.py)

### Writer 相关

- [tests/test_writer_offline.py](tests/test_writer_offline.py)（该测试期望的 URL 汇总后处理逻辑当前未实现）
