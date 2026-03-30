# CLAUDE.md — 深度研究代理：编码计划

## 项目概述

构建一个包含两个阶段的**human-in-the-loop深度研究代理**，支持将网络搜索结果直接作为上下文，或通过向量数据库进行压缩检索的多种模式。
1.  **offline ingestion** — 解析本地文档（PDF等），进行分块、嵌入，并存储在向量数据库中。
2.  **online research agent** — 接受用户查询，规划研究任务，检索证据，自我反思证据是否充分，生成Markdown报告。

### 运行模式 (Operation Modes)
代理支持以下几种操作模式：

1.  **极速网络模式 (Fast Web / Baseline)**:
    *   **流程**: `Query -> Plan -> Web Search -> **Evidence** -> Report`
    *   **原理**: 仅进行网络搜索，结果直接作为上下文用于生成报告（不经过向量库）。
    *   **适用场景**: 时效性强的新闻、无需复杂推理的简单查询。

2.  **深度压缩模式 (Deep RAG / Compression context)**:
    *   **流程**: `Query -> Plan -> (Local Search + Web Search) -> **Ingest into VectorDB** -> **Retrieve Context** -> Report`
    *   **原理**: 将网络搜索得到的大量信息作为“临时知识”摄入向量数据库。随后，利用 Query 再次从数据库中检索最相关的片段。
    *   **目的**: 通过向量相似度匹配（Re-ranking）从海量搜索结果中提取精华，实现**上下文压缩**，减少LLM上下文窗口压力，提高相关性。

3.  **纯本地模式 (Local Only)**:
    *   **流程**: `Query -> Plan -> **Local Retrieve** -> Report`
    *   **原理**: 仅检索本地已有的向量知识库，不联网。
    *   **适用场景**: 敏感数据处理、离线环境。

4.  **全自动模式 (Auto/No-HITL)**:
    *   上述任一模式均可配置是否启用 `auto_approve` 跳过人工批准环节，直接由查询生成最终报告。

---

## 架构摘要

```
                                                  [本地文档]
                                                      ↓
                                           (离线) 摄取/分块/嵌入
                                                      ↓
[用户查询] → 状态/规划 → [HITL] → 路由器 —(模式选择)——→ 向量数据库 (VectorDB)
                                    |                     ↑    ↕
                                    | (Baseline)          | (Deep RAG: 存入+检索)
                                    |                     |
                                    ↓                     |
                                网络搜索器 ———————————————┘
                                    |
                                    ↓ (Direct Context)
                                 证据融合
                                    ↓
                                 自我反思
                                    ↓
                                 报告编写
```

---

## 目录结构

```
project/
├── CLAUDE.md
├── .env                         # API密钥 (OPENAI_API_KEY, TAVILY_API_KEY, 等)
├── requirements.txt
│
├── ingestion/
│   ├── __init__.py
│   ├── parser.py                # 多模态解析器 (PDF, 图片, 表格)
│   ├── chunker.py               # 语义分块
│   ├── embedder.py              # 嵌入模型包装器
│   └── vector_store.py          # 向量数据库接口 (例如, Chroma / Qdrant)
│
├── agent/
│   ├── __init__.py
│   ├── state.py                 # 全局状态管理器 (共享状态数据类)
│   ├── planner.py               # 规划器: 分解查询 → 研究任务
│   ├── router.py                # 路由器: 本地优先, 网络备用
│   ├── retriever.py             # 向量数据库检索包装器
│   ├── web_searcher.py          # 网络搜索工具 (例如, Tavily / SerpAPI)
│   ├── evidence_fusion.py       # 合并 + 总结检索到的证据
│   ├── reflection.py            # 自我反思: 判断证据是否充分
│   ├── writer.py                # 报告编写器: 生成Markdown草稿
│   └── hitl.py                  # human-in-the-loop交互助手
│
├── graph/
│   ├── __init__.py
│   └── research_graph.py        # LangGraph / 编排图定义
│
├── ui/
│   ├── cli.py                   # 用于human-in-the-loop提示的CLI界面
│   └── app.py                   # (可选) Gradio / Streamlit UI
│
└── tests/
    ├── test_ingestion.py
    ├── test_planner.py
    ├── test_reflection.py
    └── test_end_to_end.py
```

---

## 模块规格

### 1. `ingestion/parser.py` — 多模态解析器

**目的:** 将原始本地文件（PDF, DOCX, 图片）解析为结构化文本。

**关键函数:**
```python
def parse_file(file_path: str) -> list[Document]:
    """
    检测文件类型，分派到相应的子解析器。
    返回一个包含 .page_content 和 .metadata 的 Document 对象列表。
    """
```

**实现说明:**
- 默认解析器: `pypdf` 或 `pdfplumber` (轻量, 纯文本PDF)。
- 混合内容解析: `unstructured` (表格, 图片较多的PDF)。
- OCR 方案: `pytesseract` (扫描版PDF)。
- 可选高级解析器: `MinerU` (Magic-PDF, 输出Markdown/JSON, 适合公式/表格密集文档)。
    - 当环境允许时启用 (GPU/CPU均可, 但依赖较重)。
    - 将 `MinerU` 作为可插拔解析器, 通过配置切换。
- 每个 `Document` 都应携带元数据: `source`, `page`, `file_type`。

---

### 2. `ingestion/chunker.py` — 语义分块器

**目的:** 将文档分割成语义连贯的块以便嵌入。

**关键函数:**
```python
def chunk_documents(docs: list[Document], chunk_size: int = 512, overlap: int = 64) -> list[Document]:
    """
    应用具有语义感知的递归字符分割。
    返回保留元数据的较小Document块。
    """
```

**实现说明:**
- 使用 `langchain.text_splitter.RecursiveCharacterTextSplitter`。
- 优先在段落/句子边界进行分割。
- 在元数据中维护 `chunk_index` 以便追溯。

---

### 3. `ingestion/embedder.py` — 嵌入器

**目的:** 将文本块转换为向量嵌入。

**关键函数:**
```python
def embed_documents(chunks: list[Document]) -> list[tuple[Document, list[float]]]:
    """嵌入每个块；返回 (document, embedding) 对的列表。"""
```

**实现说明:**
- 默认模型: `text-embedding-v1` (OpenAI)
- 批量进行嵌入调用以遵守 API 速率限制。
- 缓存嵌入以避免冗余的重新嵌入。

---

### 4. `ingestion/vector_store.py` — 向量数据库接口

**目的:** 持久化和查询嵌入。

**关键函数:**
```python
def upsert(texts: list[str], embeddings: list[list[float]], metadatas: list[dict] | None = None) -> int: ...
def similarity_search(query_embedding: list[float], top_k: int = 5) -> list[Document]: ...
```

**实现说明:**
- 使用 `chromadb` (本地, 易于设置) 或 `qdrant-client` (生产就绪)。
- 集合名称应可通过 `.env` 配置。
- VectorStore 只负责向量持久化与检索；嵌入由 `embedder` 完成。
- 包括一个 `reset_collection()` 工具用于开发重置。

---

### 5. `agent/state.py` — 全局状态管理器

**目的:** 在图中每个节点间传递的中央共享状态。

```python
from dataclasses import dataclass, field

@dataclass
class ResearchState:
    query: str
    mode: str = "deep_rag"  # 模式: "fast_web", "deep_rag", "local_only"
    auto_approve: bool = False  # 是否跳过HITL批准
    research_tasks: list[str] = field(default_factory=list)
    plan_approved: bool = False
    retrieved_evidence: list[Document] = field(default_factory=list)
    reflection_iterations: int = 0
    max_reflection_iterations: int = 3
    report_draft: str = ""
    report_approved: bool = False
    final_report: str = ""
    messages: list[dict] = field(default_factory=list)  # 对话历史
```

---

### 6. `agent/planner.py` — 规划器

**目的:** 将用户查询分解为一系列具体的研究子任务。

**关键函数:**
```python
def plan(state: ResearchState) -> ResearchState:
    """
    使用用户查询 + 可选的用户反馈调用LLM。
    更新 state.research_tasks。
    """
```

**提示模板:**
```
你是一个研究规划助理。
给定用户查询: "{query}"
{可选: "用户对先前计划的反馈: {feedback}"}

将此分解为3–7个具体的、可搜索的子问题。
返回一个编号列表。要求精确且不重叠。
```

---

### 7. `agent/hitl.py` — human-in-the-loop

**目的:** 暂停执行并收集人类的批准或反馈。

**关键函数:**
```python
def review_plan(state: ResearchState) -> tuple[bool, str]:
    """
    向用户显示研究任务。
    返回 (approved: bool, feedback: str)。
    """

def review_report(state: ResearchState) -> tuple[bool, str]:
    """
    显示Markdown报告草稿。
    返回 (approved: bool, feedback: str)。
    """
```

**实现说明:**
- CLI模式: `input()` 提示，带有 Y/n + 可选的自由文本反馈。
- UI模式: Gradio按钮回调或Streamlit `st.button`。
- 反馈字符串在重新循环时传回给规划器或编写器。

---

### 8. `agent/router.py` — 路由器

**目的:** 根据运行模式 (`state.mode`) 决定检索路径和数据流向。

**关键函数:**
```python
def route(state: ResearchState) -> str:
    """
    根据模式决定数据流向:
    - 'fast_web': WebSearcher --> EvidenceFusion
    - 'deep_rag': WebSearcher --> VectorDB --> Retriever --> EvidenceFusion
    - 'local_only': Retriever --> EvidenceFusion
    """
```

**逻辑与上下文压缩:**
1.  **Fast Web (Baseline)**: 
    - 路径: 调用 `web_searcher`。
    - 数据: 搜索引擎结果直接写入 `state.retrieved_evidence`。无压缩。
2.  **Deep RAG (Context Compression)**: 
    - 路径: 先调用 `web_searcher`，然后将结果传递给向量库摄取模块，最后调用 `retriever`。
    - **压缩原理**: 利用 Embedding 模型将非结构化的网页内容向量化，通过与 Query 的向量相似度匹配，仅提取 Top-K 相关片段作为最终上下文，去除无关噪音。
3.  **Local Only**: 
    - 路径: 直接调用 `retriever`。

---

### 9. `agent/web_searcher.py` — 网络搜索器

**目的:** 从网络获取实时信息。

**关键函数:**
```python
def search(query: str, num_results: int = 5) -> list[Document]:
    """将网络片段作为带有URL元数据的Document对象返回。"""
```

**实现说明:**
- 使用 **Tavily API** (`tavily-python`) 进行研究级搜索。
- 备用方案: `duckduckgo-search` (无需API密钥)。
- 去除HTML，截断长页面，在元数据中保留 `url`。

---

### 10. `agent/evidence_fusion.py` — 证据融合与总结器

**目的:** 将从本地+网络检索到的文档合并成一个连贯的证据摘要。

**关键函数:**
```python
def fuse(state: ResearchState) -> ResearchState:
    """
    获取 state.retrieved_evidence, 调用LLM为每个子任务进行总结。
    将结构化摘要附加到状态中。
    """
```

**提示模板:**
```
你是一名研究分析师。
子任务: "{task}"
证据:
{evidence_chunks}

总结与子任务相关的关键发现。
通过 [source_index] 引用来源。要求简洁但详尽。
```

---

### 11. `agent/reflection.py` — 自我反思

**目的:** 判断当前证据是否足以回答原始查询。

**关键函数:**
```python
def reflect(state: ResearchState) -> str:
    """
    返回 'sufficient' 或 'insufficient'。
    如果不足，生成一个新的澄清性子查询并附加到 research_tasks。
    """
```

**提示模板:**
```
原始查询: "{query}"
到目前为止的证据摘要:
{summary}

证据是否足以撰写一份全面、准确的报告？
如果否，输出: INSUFFICIENT: <new_search_query>
如果是，输出: SUFFICIENT
```

**循环保护:** 在重新路由之前检查 `state.reflection_iterations < state.max_reflection_iterations` 以防止无限循环。

---

### 12. `agent/writer.py` — 报告编写器

**目的:** 生成结构化的Markdown研究报告。

**关键函数:**
```python
def write(state: ResearchState) -> ResearchState:
    """从证据摘要生成 state.report_draft。"""
```

**输出格式:**
```markdown
# 深度研究报告: {query}

## 执行摘要
...

## 关键发现
### 发现 1: ...
### 发现 2: ...

## 证据与来源
- [1] ...

## 结论
...

## 局限性与进一步研究
...
```

---

### 13. `graph/research_graph.py` — 编排图

**目的:** 使用LangGraph（或一个简单的状态机）将所有节点连接在一起。

```python
from langgraph.graph import StateGraph

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner.plan)
    graph.add_node("hitl_plan", hitl.review_plan)
    graph.add_node("router", router.route)
    graph.add_node("retriever", retriever.retrieve)
    graph.add_node("web_searcher", web_searcher.search)
    graph.add_node("ingester", ingestion.ingest)  # 新增: Web到VectorDB的桥接
    graph.add_node("evidence_fusion", evidence_fusion.fuse)
    graph.add_node("reflection", reflection.reflect)
    graph.add_node("writer", writer.write)
    graph.add_node("hitl_report", hitl.review_report)

    # 边与逻辑
    graph.set_entry_point("planner")
    
    # 自动模式跳过计划批准
    graph.add_conditional_edges("planner", lambda s: "router" if s.auto_approve else "hitl_plan")
    graph.add_conditional_edges("hitl_plan", lambda s: "planner" if not s.plan_approved else "router")

    # 路由逻辑 (基于模式)
    def route_decision(state):
        if state.mode == "fast_web":
            return "web_searcher"        # Web -> Evidence
        elif state.mode == "deep_rag":
            return ["web_searcher", "retriever"] # 并行: Deep RAG通常需要Web结果先入库再Retrieve，但在Graph中可并行或作为Pipeline
        elif state.mode == "local_only":
            return "retriever"

    # Deep RAG 特殊管道: Web -> Ingest -> Retrieve
    # 这里为了简化，假设 Web Searcher 如果在 Deep RAG 模式下，会自动调用 Ingestion 或连接到 Ingestion 节点
    # 实际 Graph 可能需要更精细的边:
    # WebSearcher -> (if deep_rag) -> Ingester -> Retriever -> EvidenceFusion
    # WebSearcher -> (if fast_web) -> EvidenceFusion
    
    graph.add_edge("web_searcher", "evidence_fusion") # Fast Web Path
    # (Deep RAG逻辑需在 Web 节点内部或通过 Condition Edge 处理流入 Ingester)

    graph.add_edge("retriever", "evidence_fusion")
    graph.add_edge("evidence_fusion", "reflection")
    graph.add_conditional_edges("reflection", lambda s: "router" if s.needs_more else "writer")
    
    graph.add_conditional_edges("writer", lambda s: END if s.auto_approve else "hitl_report")
    graph.add_conditional_edges("hitl_report", lambda s: "planner" if not s.report_approved else END)

    return graph.compile()
```

---

## 环境变量 (`.env`)

```env
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small

# Vector DB
CHROMA_COLLECTION=research_kb
CHROMA_PERSIST_DIR=./chroma_db

# Web Search
WEB_SEARCH_PROVIDER=tavily    # tavily | duckduckgo
TAVILY_API_KEY=tvly-...

# Parser
PARSER_PROVIDER=pypdf         # pypdf | mineru
MINERU_MODEL_DIR=~/.cache/mineru

# Agent Config
MAX_REFLECTION_ITERATIONS=3
LOCAL_RETRIEVAL_TOP_K=5
WEB_SEARCH_RESULTS=5
```

---

## 依赖项 (`requirements.txt`)

```
# LLM & Orchestration
openai>=1.0
langchain>=0.2
langgraph>=0.1
langchain-openai

# Document Parsing
pypdf
pdfplumber
unstructured[pdf]
pytesseract        # 可选 OCR
mineru             # 可选: MinerU (重依赖, 适合复杂PDF)

# Vector DB
chromadb

# Web Search
tavily-python
duckduckgo-search  # 备用

# Utilities
python-dotenv
pydantic>=2.0
rich               # 漂亮的CLI输出

# UI (可选)
gradio
streamlit

# Testing
pytest
pytest-asyncio
```

---

## 逐步构建计划 (Step-by-Step Build Plan)

本构建计划按功能分层，旨在通过一系列增量步骤，从最核心的无状态搜索开始，逐步构建起支持深度RAG和HITL的复杂系统。每一步完成后，都应运行相应的测试以确保基座稳固。

### 阶段一：极速网络基线 (Fast Web Baseline)
**目标**: 实现一个无状态的 Web Search Agent，能够接收 Query 并生成报告。

1.  **基础设施 (Setup)**
    *   **Action**: 设置项目结构，配置 `.env`，创建 `agent/state.py` 定义 `ResearchState`。
    *   **Validation**: 运行简单的脚本打印 `ResearchState` 实例，确认环境配置正确。

2.  **网络搜索器 (Web Searcher)**
    *   **Action**: 实现 `agent/web_searcher.py`，集成 Tavily API。
    *   **Validation**: 编写 `tests/test_search.py`，输入 "AI Agent trends 2024"，验证返回非空的 `Document` 列表（含 URL 元数据）。

3.  **规划与融合 (Planner & Fusion)**
    *   **Action**: 实现 `agent/planner.py` (Query 解构) 和 `agent/evidence_fusion.py` (LLM 总结)。
    *   **Validation**: 编写 `tests/test_planner.py` 验证生成的子任务列表；编写脚本手动输入模拟的 Document 列表，验证 Fusion 生成的摘要质量。

4.  **端到端管道 V1 (Pipeline - Fast Web)**
    *   **Action**: 在 `graph/research_graph.py` 中构建最小 MVP 图：`Planner -> WebSearcher -> EvidenceFusion -> Writer`。暂时跳过 HITL 和 Router。
    *   **Validation**: 运行 `ui/cli.py` (需创建基础版本)，输入 Query，检查是否生成了 Markdown 报告。

### 阶段二：本地知识库 (Local Knowledge Base)
**目标**: 建立离线数据的摄取和检索能力，为 RAG 打基础。

5.  **文档摄取 (Ingestion Core)**
    *   **Action**: 实现 `ingestion/parser.py` (PDF 解析) 和 `ingestion/chunker.py` (语义切分)。
    *   **Validation**: 放置一个测试 PDF，运行 `tests/test_parser.py`，检查输出的 Chunks 是否完整且语义连贯。

6.  **向量存储 (Vector Store)**
    *   **Action**: 实现 `ingestion/embedder.py` 和 `ingestion/vector_store.py` (ChromaDB)。实现 `upsert` 和 `query` 接口。
    *   **Validation**: 编写 `tests/test_vector_db.py`。
        - 步骤: 1. 清空 DB; 2. 插入测试文本块; 3. Query 相关关键词; 4. 验证返回结果是刚才插入的块。

7.  **纯本地管道 V2 (Pipeline - Local Only)**
    *   **Action**: 实现 `agent/retriever.py`。在 Graph 中添加 `Retriever` 节点。更新 `Router` 支持 `state.mode="local_only"`。
    *   **Validation**: 设置 `mode="local_only"`。在 `ingestion/` 下预置一些独特内容的 PDF。提问相关问题，验证 Agent 仅使用本地知识回答。

### 阶段三：深度 RAG 与上下文压缩 (Deep RAG & Context Compression)
**目标**: 将网络搜索结果“压缩”进向量库，实现高相关性的上下文提取。

8.  **即时摄取 (On-the-fly Ingestion)**
    *   **Action**: 创建 `ingestion/bridge.py` 或在 `ingestion/__init__.py` 中暴露函数，允许将 `List[Document]` (来自 Web) 直接通过 Embedding 存入 VectorDB（临时 Collection 或打上 Session Tag）。
    *   **Validation**: 编写测试：搜索 Web -> 调用 Ingestion 函数 -> 立即查询 VectorDB -> 确认能查到刚才的 Web 内容。

9.  **路由与图重构 (Router Refactor)**
    *   **Action**: 更新 `agent/router.py` 和 `graph/research_graph.py`。
        - 逻辑：当 `mode="deep_rag"` 时，路径变为 `WebSearcher -> Ingester -> Retriever -> EvidenceFusion`。
    *   **Validation**: 运行 `tests/test_deep_rag.py`。验证流程日志：确认执行了网络搜索，确认执行了写入 DB 操作，确认最终 Retrieve 到了数据。对比 "Fast Web" 和 "Deep RAG" 对同一复杂问题的回答质量（可选）。

### 阶段四：人机回圈与完善 (HITL & Polish)
**目标**: 增加人类控制点，完善系统稳定性。

10. **交互节点 (HITL Nodes)**
    *   **Action**: 实现 `agent/hitl.py`。在 Graph 中插入 `hitl_plan` 和 `hitl_report` 节点。实现 `auto_approve` 开关逻辑。
    *   **Validation**: 运行 CLI。
        - 测试场景 A: 拒绝计划，提供反馈 -> 确认 Planner 生成了新计划。
        - 测试场景 B: 开启 `auto_approve=True`，确认全程无中断。

11. **自我反思 (Self-Reflection)**
    *   **Action**: 实现 `agent/reflection.py`。设置 `MAX_REFLECTION_ITERATIONS`。
    *   **Validation**: 构造一个“信息不足”的测试桩（Mock Retriever 返回空），确认 Graph 进入了 Reflection 循环并生成了新的搜索 Query。

---

## 关键设计约束

- **本地优先检索:** 除非是极速网络模式，否则在发出网络请求之前总是检查向量数据库。
- **huamn in the loop是阻塞的:** 代理必须在两个检查点（计划批准和报告批准）完全暂停并等待人类输入。不要自动批准。
- **循环安全:** 反思循环的上限为 `MAX_REFLECTION_ITERATIONS`。达到上限后，无论如何都继续进行写作。
- **无状态节点:** 每个图节点接收完整的 `ResearchState` 并返回一个更新后的副本。没有全局突变。
- **结构化输出:** 规划器和反思节点应使用 `response_format={"type": "json_object"}` 或Pydantic输出解析器以确保可靠性。
- **来源可追溯性:** 存储在状态中的每个证据块都必须携带其来源（文件路径或URL）。最终报告必须引用来源。

---

## 测试检验清单 (Test Verification Checklist)

本检验清单配合逐步构建计划，确保每一阶段交付都是可用的。

1.  ### 基础架构与网络基线
    *   [ ] **State Init**: `agent/state.py` 实例化无错。
    *   [ ] **Web Search**: `tests/test_web_searcher.py` 能够返回 Google/Tavily 搜索结果 (check URL field)。
    *   [ ] **Plan Generation**: `tests/test_planner.py` 输入 "Quantum Computing" 返回 3-5 个结构化子任务。
    *   [ ] **Fast Web E2E**: 运行 `ui/cli.py` (Fast Web mode)，验证报告是否包含最新的网络信息。

2.  ### 本地知识库与检索
    *   [ ] **Parser Quality**: `tests/test_parser.py` 解析测试 PDF，页数正确，文本无乱码。
    *   [ ] **Vector DB CRUD**: `tests/test_vector_db.py` 插入 -> 查询 -> 删除，基本功能可用。
    *   [ ] **Local RAG E2E**: 提供独特内容的本地 PDF，设置 `mode="local_only"`，验证报告是否准确引用了该内容。

3.  ### 深度 RAG (关键特性)
    *   [ ] **Ingest Bridge**: 能够将 List[Document] (来自Web) 实时转为 Embeddings 并存入指定 Collection。
    *   [ ] **Compression Effect**: 打印 "Deep RAG" 模式中间过程：Web Search (例如20条) -> Ingest -> Retrieve (例如Top-5)。验证 Retrieve 出的 Top-5 是否确实是 Top-20 中最相关的部分。
    *   [ ] **Mode Switching**: 设置不同 Mode，观察日志，确认 Pipeline 路径切换正确 (Fast Web 跳过 Ingest/Retrieve; Deep RAG 经过 Ingest/Retrieve)。

4.  ### 交互与鲁棒性
    *   [ ] **HITL Interruption**: 在 Plan 阶段拒绝，提供 "请聚焦于 X 方面"，验证 Planner 生成了包含 X 的新计划。
    *   [ ] **Reflection Loop**: 模拟 "Evidence Insufficient"，观察系统是否自动生成了额外的搜索 Query 并执行了第二轮循环。
    *   [ ] **Final Report**: 检查最终 Markdown 报告格式，是否包含引用来源 `[1]` 等标记。
