# 深度研究代理（dr_agent）

本项目是一个面向研究类任务的智能体系统，支持本地文档与网络信息的联合检索、证据汇总与报告生成。

## 1. 已实现系统功能

- 多模式运行
	- fast_web：规划后直接联网检索并生成报告。
	- deep_rag：联网检索后写入向量库，再检索压缩上下文生成报告。
	- hybrid_deep_rag：先摄取本地文件，再走 planner -> websearcher，将网页内容写入同一 Chroma 集合后统一检索生成。

- hybrid_deep_rag 本地证据保留策略
	- 无论检索相关性如何，都会保留本地证据并并入最终证据集。
	- 报告参考来源中会展示本地文档名（例如：本地文档: example.pdf）。

- 检索能力
	- 支持 dense、bm25、hybrid_rrf 三种检索策略（通过环境变量切换）。
	- 支持按子任务多轮检索与去重融合。

- 本地文档处理
	- 支持 PDF、TXT、MD 解析。
	- 支持语义分块、向量化、写入 Chroma 持久化库。
	- 支持独立摄取脚本 [ingest_local.py](ingest_local.py)。

- 报告生成
	- 输出结构化 Markdown 报告。
	- 支持证据编号引用与去重后的参考来源列表。

## 2. 核心模块

- 状态管理
	- [agent/state.py](agent/state.py)：维护 query、mode、local_files、retrieved_evidence、report_draft 等运行状态。

- 流程编排
	- [graph/research_graph.py](graph/research_graph.py)：定义主流程节点与路由逻辑，包含 local_ingester、web_search、ingester、retriever、writer。

- 任务规划
	- [agent/planner.py](agent/planner.py)：将用户问题拆解为可执行子任务。

- 网络搜索
	- [agent/web_searcher.py](agent/web_searcher.py)：支持 Tavily 与 DuckDuckGo 两种搜索后端。

- 检索与融合
	- [agent/retriever.py](agent/retriever.py)：向量检索、BM25 检索、RRF 融合检索。
	- [agent/sparse_bm25.py](agent/sparse_bm25.py)：BM25 索引与检索工具。

- 报告写作
	- [agent/writer.py](agent/writer.py)：基于证据生成报告，并构建网页与本地文档引用列表。

- 数据摄取与向量库
	- [ingestion/parser.py](ingestion/parser.py)：文档解析。
	- [ingestion/chunker.py](ingestion/chunker.py)：语义分块。
	- [ingestion/embedder.py](ingestion/embedder.py)：支持 API 嵌入和本地嵌入模型。
	- [ingestion/vector_store.py](ingestion/vector_store.py)：Chroma 封装与检索接口。

- CLI 入口
	- [ui/cli.py](ui/cli.py)：命令行参数解析与流程启动。

## 3. 项目文件结构

主要目录结构如下（省略缓存与产物目录中的大量文件）：

dr_agent/
├── README.md
├── requirements.txt
├── .env.example
├── ingest_local.py
├── agent/
│   ├── state.py
│   ├── planner.py
│   ├── web_searcher.py
│   ├── retriever.py
│   ├── sparse_bm25.py
│   ├── writer.py
│   ├── router.py
│   └── evidence_fusion.py
├── graph/
│   ├── research_graph.py
│   └── split_graph.py
├── ingestion/
│   ├── parser.py
│   ├── chunker.py
│   ├── embedder.py
│   └── vector_store.py
├── ui/
│   └── cli.py
├── tests/
│   ├── test_end_to_end.py
│   ├── test_deep_rag.py
│   ├── test_hybrid_deep_rag.py
│   ├── test_mode_comparison.py
│   ├── test_writer_reference_local.py
│   └── ...
├── eval/
├── chroma_db/
└── reports/

## 4. 环境要求

- Python
	- 建议 Python 3.10 及以上。
	- 建议使用虚拟环境。

- 依赖安装
	- 在项目根目录执行：
		- pip install -r requirements.txt

- 关键环境变量
	- LLM
		- OPENAI_API_KEY（必填）
		- OPENAI_MODEL（可选）
		- OPENAI_BASE_URL（可选）
	- Web Search
		- WEB_SEARCH_PROVIDER=tavily 或 duckduckgo
		- TAVILY_API_KEY（当 provider=tavily 时必填）
	- Embedding
		- EMBEDDING_MODEL（API 模式）
		- EMBEDDING_MODEL_PATH、EMBEDDING_DEVICE、EMBEDDING_LOCAL_BATCH_SIZE（本地模型模式）
		- EMBEDDING_MAX_WORKERS、EMBEDDING_PARALLEL（并行参数）
	- Vector DB
		- CHROMA_COLLECTION（可选）
		- CHROMA_PERSIST_DIR（可选）
	- Retrieval
		- LOCAL_RETRIEVAL_TOP_K
		- RETRIEVAL_MODE（dense | bm25 | hybrid_rrf）
		- RRF_K

- 运行示例
	- hybrid_deep_rag：
		- python -m ui.cli --mode hybrid_deep_rag --query "deep research agent" --local-files ./tests/example.pdf --output ./reports/
    - deep_rag:
        - python -m ui.cli --mode deep_rag --query "deeep research agent" --output ./reports/
