# 深度研究代理：实现概览与规格（基于当前代码）

本文以当前目录 `dr_agent/` 下的实际实现为准（`agent/`、`graph/`、`ingestion/`、`ui/`、`tests/`）。

## 1. 实现状态总览（What’s Done）

| 能力/模式 | 状态 | 实现位置（核心） | 说明 |
|---|---:|---|---|
| Fast Web（极速网络基线） | ✅ 已实现 | `graph/research_graph.py`、`agent/web_searcher.py` | 子任务逐个搜索 → 汇总证据 → 写报告 |
| Deep RAG（网络结果入库→向量检索压缩） | ✅ 已实现 | `graph/research_graph.py`（`ingester`、`retriever`） | Web 结果写入 Chroma 临时集合，再按子任务 Top-K 检索 |
| Local Only（纯本地向量检索） | ✅ 已实现（但 CLI 未暴露） | `graph/research_graph.py`、`agent/retriever.py` | 仅检索本地知识库，无网络 |
| Test Mode（复用 Web 搜索结果） | ✅ 已实现 | `graph/research_graph.py`（`test_mode`/`web_search_cache`） | Fast Web 先填充缓存，Deep RAG 复用缓存避免随机性 |
| 离线入库（本地文档→分块→嵌入→Chroma） | ✅ 已实现 | `ingest_local.py`、`ingestion/*` | 支持 `.pdf/.txt/.md` |
| 报告生成（Markdown） | ✅ 已实现 | `agent/writer.py` | 基于“证据摘要”或原始证据写报告 |
| HITL（人机回圈：计划/报告审批） | ⬜ 未实现（仅预留字段） | `agent/state.py` | 目前没有 `hitl` 节点/模块 |
| Reflection（自我反思循环补搜） | ⬜ 未实现（仅预留字段） | `agent/state.py` | 目前没有 reflection 节点与循环逻辑 |
| 证据来源 URL 自动汇总到报告末尾 | ⬜ 未实现（测试期望存在） | `tests/test_writer_offline.py` | Writer 当前不会生成“URL 列表/映射” |

## 2. 快速运行（CLI / 入库）

### 2.1 运行在线研究（Fast Web / Deep RAG）

入口：`ui/cli.py`

```bash
# Fast Web（默认）
python -m ui.cli --mode fast_web --query "AI Agent 趋势" --output reports/

# Deep RAG（会把 Web 结果写入临时集合后再检索）
python -m ui.cli --mode deep_rag --query "AI Agent 趋势" --output reports/

# Test Mode：复用 Web 搜索结果（先 Fast Web 填缓存，再把缓存传给 Deep RAG，见 tests 用法）
python -m ui.cli --mode fast_web --test-mode --query "Python 异常处理" --output reports/
```

说明：
- Web provider 由 `WEB_SEARCH_PROVIDER` 决定：`tavily` 或 `duckduckgo`。
- `duckduckgo` 需要安装 `ddgs`（已在 `requirements.txt` 中声明）。

### 2.2 离线入库本地文档（Local KB）

入口：`ingest_local.py`

```bash
# 摄取单个文件
python ingest_local.py ./docs/sample.pdf --collection research_kb

# 摄取整个目录（默认递归）
python ingest_local.py ./docs/ --collection research_kb
```

## 3. 模式与数据流（How It Works）

### 3.1 统一数据结构

- 文档统一用 `langchain_core.documents.Document` 表示：
  - `page_content`: 文本
  - `metadata`: 元信息（`url/title/source/...` 或 `filename/page/...`）
- 图节点之间共享状态 `ResearchState`（见 `agent/state.py`）：
  - 输入：`query`、`mode`、`test_mode`、`web_search_cache`
  - 中间结果：`research_tasks`、`retrieved_evidence`、`messages`、`report_draft`
  - 预留但未启用：`hitl_enabled`、`reflection_iterations` 等

### 3.2 Fast Web（已实现）

流程：

`Query → Planner → WebSearcher → EvidenceFusion → Writer → Report`

要点：
- Planner 生成 3–7 个可搜索子问题（写入 `state.research_tasks`）。
- WebSearcher 针对每个子任务搜索，并把所有搜索结果合并为 `state.retrieved_evidence`。
- EvidenceFusion 按“子任务”对证据进行总结，把结构化摘要存入 `state.messages`（角色 `evidence_summaries`）。
- Writer 优先使用摘要写报告。

### 3.3 Deep RAG（已实现：上下文压缩）

流程：

`Query → Planner → WebSearcher → Ingester(写入 Chroma) → Retriever(Top-K) → EvidenceFusion → Writer → Report`

要点：
- Ingester 将 Web 结果补充 `ingestion_session/source` 后嵌入并写入 Chroma。
- Retriever 不直接使用全部 Web 结果，而是对每个子任务向量检索 Top-K 片段，达到“压缩上下文、提高相关性”的效果。

### 3.4 Local Only（已实现但 CLI 未暴露）

流程：

`Query → Planner → Retriever(本地集合) → EvidenceFusion → Writer → Report`

要点：
- 不触发网络搜索。
- 依赖先使用 `ingest_local.py` 把资料写入本地集合（默认 `CHROMA_COLLECTION=research_kb`）。

### 3.5 Test Mode（已实现：复用 Web 结果）

实现位置：`graph/research_graph.py` 的 `_web_search_node`

- 当 `state.test_mode=True` 且 `state.web_search_cache` 非空：直接复用缓存填充 `state.retrieved_evidence`。
- 当 `state.test_mode=True` 且缓存为空：执行搜索并把结果写入 `state.web_search_cache`。

此模式用于在对比 Fast Web vs Deep RAG 时，尽量消除“搜索结果随机性”。（见 `tests/test_mode_comparison.py`）

## 4. 目录结构（当前实际）

```
dr_agent/
├── CLAUDE.md
├── .env / .env.example
├── requirements.txt
├── ingest_local.py              # 离线入库：本地文件/目录 → Chroma
├── agent/
│   ├── state.py                 # ResearchState 数据结构
│   ├── planner.py               # 规划（LLM）
│   ├── router.py                # 路由（按 mode 决策）
│   ├── web_searcher.py          # Web 搜索（Tavily / DuckDuckGo）
│   ├── retriever.py             # 向量检索（按子任务 Top-K）
│   ├── evidence_fusion.py       # 证据融合（LLM summarization）
│   └── writer.py                # Markdown 报告生成（LLM）
├── ingestion/
│   ├── parser.py                # PDF/TXT/MD 解析
│   ├── chunker.py               # RecursiveCharacterTextSplitter 分块
│   ├── embedder.py              # OpenAI embeddings
│   └── vector_store.py          # ChromaDB 封装
├── graph/
│   └── research_graph.py        # LangGraph 编排（含 ingester/test_mode）
├── ui/
│   └── cli.py                   # CLI 入口（fast_web/deep_rag）
└── tests/
    ├── test_end_to_end.py
    ├── test_deep_rag.py
    ├── test_local_rag.py
    ├── test_mode_comparison.py
    ├── test_parser.py
    ├── test_planner.py
    ├── test_vector_db.py
    ├── test_web_searcher.py
    └── test_writer_offline.py
```

## 5. 模块规格（保留实现细节，减少代码）

> 下面每个模块都以“当前实现”为准，突出：输入/输出、关键逻辑、约束与边界。

### 5.1 `agent/state.py` — 全局状态（已实现）

- 类型：`dataclass ResearchState`
- 关键字段：
  - `query: str`
  - `mode: str`：`fast_web` / `deep_rag` / `local_only`
  - `test_mode: bool` + `web_search_cache: List[Document]`
  - `research_tasks: List[str]`
  - `retrieved_evidence: List[Document]`
  - `messages: List[dict]`：用于在节点间“传递结构化中间产物”（例如 `evidence_summaries`）
- 预留字段（目前未被图使用）：`hitl_enabled`、`plan_approved`、`reflection_iterations`、`report_approved`、`final_report` 等。

### 5.2 `agent/planner.py` — 规划器（已实现）

- 输入：`ResearchState(query, messages?)`
- 输出：更新 `state.research_tasks`
- LLM：`langchain_openai.ChatOpenAI`
  - `OPENAI_MODEL` 默认 `qwen3-max`
  - 支持 `OPENAI_BASE_URL`（便于兼容不同 OpenAI 兼容服务）
- 反馈机制：若 `state.messages` 最后一条包含 `feedback` 关键词，会把它拼到 prompt 中让模型“调整计划”。
- 解析：`_parse_planning_response` 通过正则剥离编号/项目符号，得到任务列表。

### 5.3 `agent/web_searcher.py` — 网络搜索（已实现）

- `search(query, num_results)` 根据 `WEB_SEARCH_PROVIDER` 选择实现：
  - `tavily`：需要 `TAVILY_API_KEY`
  - `duckduckgo`：使用 `ddgs`，无需 API key
- 输出：`List[Document]`，`metadata` 至少含 `url/title/source`。
- 文本清洗：去除 HTML 标签，长文本截断到 2000 字符。

### 5.4 `agent/retriever.py` — 向量检索（已实现）

- 依赖：
  - `ingestion.embedder.embed_query` 生成查询向量
  - `ingestion.vector_store.VectorStore` 做相似度检索
- 检索策略：对 `state.research_tasks` 每个任务单独检索 Top-K，然后合并到 `state.retrieved_evidence`。
- 可选：`retrieve_with_scores` 返回 `(Document, score)`。

### 5.5 `agent/evidence_fusion.py` — 证据融合/摘要（已实现）

- 输入：
  - `state.research_tasks`
  - `state.retrieved_evidence`（来自 Web 或 Retriever）
- 输出：不直接写入 `state.report_draft`，而是把“按子任务的摘要列表”放入 `state.messages`：
  - 角色 `evidence_summaries`，内容形如 `[{"task":..., "summary":...}, ...]`
- 证据格式：会把 evidence 编号为 `[1]...[N]`，要求模型用 `[source_index]` 引用来源。

### 5.6 `agent/writer.py` — 报告撰写（已实现，但来源汇总未完善）

- 输入：`state.query` +（优先）`state.messages` 中的 `evidence_summaries`
- 若没有摘要：退回使用原始 `state.retrieved_evidence`。
- 输出：`state.report_draft`（Markdown 文本）。
- 当前限制：不会把 `Document.metadata["url"]` 自动整理成“证据与来源 URL 列表”。

### 5.7 `graph/research_graph.py` — LangGraph 编排（已实现）

- 节点：`planner` → `router` →（按 mode）→ `web_searcher` / `retriever` / `ingester` → `evidence_fusion` → `writer`
- Deep RAG 的 ingestion：
  - 会把 Web 结果写入 Chroma 的集合 `temp_web_results`（固定名称）。
  - 每条证据会被追加 `ingestion_session`（基于 query 前 20 字 + idx）。
- Deep RAG 的检索：
  - 通过“临时覆写 `agent.retriever.get_retriever`”的方式，让 Retriever 指向 `temp_web_results` 集合。
- Test Mode：`_web_search_node` 支持缓存复用（见上文）。

### 5.8 `ingestion/parser.py` — 文档解析（已实现：PDF/TXT/MD）

- 支持：
  - `.pdf`：优先 `pypdf`，失败回退 `pdfplumber`
  - `.txt/.md`：按空行分段
- 输出：`List[Document]`，metadata 包含：
  - PDF：`source/filename/page/page_count/file_type`
  - 文本：`source/filename/paragraph/file_type`
- 当前边界：未实现 docx/图片/OCR/表格结构化（文档不再宣称“多模态”）。

### 5.9 `ingestion/chunker.py` — 分块（已实现）

- 实现：`RecursiveCharacterTextSplitter`
- separators 包含中文标点（`。！？`）与英文标点，尽量在语义边界切分。
- 每个 chunk 的 metadata 追加：`chunk_index/parent_doc_index/chunk_count`。

### 5.10 `ingestion/embedder.py` — 嵌入（已实现）

- 实现：OpenAI `embeddings.create`
- 配置：
  - `EMBEDDING_MODEL`（默认 `text-embedding-v1`）
  - `OPENAI_API_KEY`、`OPENAI_BASE_URL`
- 优化：
  - `embed_text` 使用 `lru_cache(maxsize=1000)`（对重复文本有效）
  - `embed_documents` 支持 batch（默认 10）

### 5.11 `ingestion/vector_store.py` — Chroma 封装（已实现）

- 默认配置：
  - `CHROMA_COLLECTION`（默认 `research_kb`）
  - `CHROMA_PERSIST_DIR`（默认 `./chroma_db`）
- `upsert(...)` 目前使用 `collection.add(...)`（更接近“append”语义；重复写入可能产生重复记录）。
- `similarity_search(...)` / `similarity_search_with_score(...)` 使用 `collection.query`；分数来自 `distances`（越小越相似）。
- 工具方法：`reset_collection()`、`count()`、`get_all_documents()`。

## 6. 环境变量（与当前实现一致）

最小可运行（在线研究 + embeddings）：

```env
# LLM / Embeddings（两者都依赖 OpenAI 兼容接口）
OPENAI_API_KEY=...
OPENAI_BASE_URL=...            # 可选
OPENAI_MODEL=qwen3-max         # 可选，默认 qwen3-max
EMBEDDING_MODEL=text-embedding-v1  # 可选

# Web Search
WEB_SEARCH_PROVIDER=tavily     # tavily | duckduckgo
TAVILY_API_KEY=...             # 仅 tavily 需要

# Vector DB（本地知识库/持久化）
CHROMA_COLLECTION=research_kb  # 可选
CHROMA_PERSIST_DIR=./chroma_db # 可选

# Retrieval
LOCAL_RETRIEVAL_TOP_K=5        # 可选
```

## 7. 逐步构建计划（标注完成度）

> 注：这里的“已完成/未完成”以当前代码仓库为准；如果你后续补齐 HITL/Reflection，再把对应项改为 ✅。

### 阶段一：极速网络基线（Fast Web Baseline）

1. ✅ 基础设施：状态结构与环境加载（`agent/state.py` / `python-dotenv`）
2. ✅ 网络搜索器：Tavily + DuckDuckGo provider（`agent/web_searcher.py`）
3. ✅ 规划与融合：Planner + EvidenceFusion（`agent/planner.py` / `agent/evidence_fusion.py`）
4. ✅ 端到端管道：LangGraph 编排并可运行 CLI（`graph/research_graph.py` / `ui/cli.py`）

### 阶段二：本地知识库（Local KB）

5. ✅ 文档解析与分块：PDF/TXT/MD（`ingestion/parser.py` / `ingestion/chunker.py`）
6. ✅ 嵌入与向量库：OpenAI embeddings + Chroma（`ingestion/embedder.py` / `ingestion/vector_store.py`）
7. ✅ Local Only 管道（实现已具备；⬜ CLI 参数未暴露）

### 阶段三：Deep RAG（Web→入库→检索压缩）

8. ✅ 即时摄取：图内 `ingester` 写入临时集合（`graph/research_graph.py`）
9. ✅ 路由与图：按 `mode` 切换三条路径（`graph/research_graph.py` / `agent/router.py`）

### 阶段四：Test Mode（复用 Web 结果）

10. ✅ `test_mode` + `web_search_cache` 复用逻辑（`graph/research_graph.py`）

### 阶段五：交互与反思（HITL / Reflection）

11. ⬜ HITL 节点（计划/报告审批）
12. ⬜ Reflection 循环（证据不足→补充检索→再写作）
13. ⬜ 报告末尾“证据与来源 URL”自动汇总与去重

## 8. 测试检验清单（标注完成度）

> 说明：仓库里同时有“纯单元/离线测试”和“依赖真实 LLM/Web 的 e2e 测试”。

### 8.1 解析与入库（离线可跑）

- ✅ `tests/test_parser.py`：文本解析、异常分支、批量解析
- ✅ `tests/test_vector_db.py`：Chroma CRUD、相似度搜索、过滤、reset

### 8.2 Web 搜索（可能受环境影响）

- ✅ `tests/test_web_searcher.py`：Tavily key 缺失行为、DuckDuckGo 分支（依赖 `ddgs`）

### 8.3 图与端到端（依赖 OPENAI_API_KEY；部分还依赖 TAVILY_API_KEY）

- ✅ `tests/test_end_to_end.py`：Fast Web e2e（默认 duckduckgo，且需要 LLM）
- ✅ `tests/test_deep_rag.py`：Deep RAG e2e + ingestion bridge + compression effect（需要 LLM/embeddings）
- ✅ `tests/test_local_rag.py`：Local Only e2e（使用临时向量库）
- ✅ `tests/test_mode_comparison.py`：Fast Web vs Deep RAG 对比 + Test Mode 复用缓存

### 8.4 Writer 的来源后处理（当前实现未覆盖）

- ⬜ `tests/test_writer_offline.py`：期望 Writer 能根据 `evidence_index_map` 把 URL 写入“证据与来源”。目前 `agent/writer.py` 不会生成该映射，也不会做该后处理。

## 9. 已知限制与下一步（建议）

1. Deep RAG 临时集合名固定为 `temp_web_results`，且 `upsert` 会累积；建议增加“按 session 过滤/清理”的策略。
2. 修改evidence fusion提示词和路径
3. Local Only 未在 CLI 参数中提供；如要面向使用者，建议在 `ui/cli.py` 的 `--mode` choices 中加入 `local_only`。
4. HITL/Reflection 都已在 `ResearchState` 中预留字段，但图中尚无节点；补齐后可显著增强“可控性”和“覆盖不足时的自我修复”。
