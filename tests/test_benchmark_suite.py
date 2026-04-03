"""
基准测试套件：Fast Web vs Deep RAG 全面对比测试

核心测试方法：
使用 test_mode + web_search_cache 固定前半段流程（Planner -> WebSearcher），
确保两种模式使用完全相同的搜索子任务和检索到的网页内容，
仅对比后半段生成报告的质量和速度差异。

测试架构：
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 固定前半段（复用已有 test_mode 实现）                    │
│  ├─ Step 1: Fast Web 模式运行，填充 web_search_cache            │
│  └─ Step 2: 提取并保存中间状态 (tasks, cache, evidence)           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: 对比后半段（使用固定输入）                              │
│  ├─ Path A: Fast Web (EvidenceFusion -> Writer)                  │
│  │           直接使用原始 evidence                               │
│  └─ Path B: Deep RAG (Ingester -> Retriever -> Writer)         │
│             使用相同的 cache 重新摄入和检索                       │
└─────────────────────────────────────────────────────────────────┘
"""

import os
import json
import time
import tempfile
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

import pytest

from agent.state import ResearchState
from graph.research_graph import build_graph


# =============================================================================
# 测试集定义
# =============================================================================

@dataclass
class TestQuery:
    """测试查询定义"""
    id: str
    query: str
    category: str  # factual, analytical, comparative, temporal, technical
    expected_topics: List[str]  # 期望报告中包含的关键主题
    difficulty: str  # easy, medium, hard


# 预定义测试集（50个查询，覆盖5种类型）
BENCHMARK_QUERIES: List[TestQuery] = [
    # === 事实类查询 (Factual) - 10个 ===
    TestQuery("F1", "Python 列表推导式的语法是什么？", "factual",
              ["列表推导式", "语法", "for"], "easy"),
    TestQuery("F2", "什么是 RESTful API 设计原则？", "factual",
              ["REST", "API", "HTTP", "资源"], "easy"),
    TestQuery("F3", "Docker 容器和虚拟机有什么区别？", "factual",
              ["Docker", "容器", "虚拟机", "区别"], "medium"),
    TestQuery("F4", "什么是神经网络中的反向传播算法？", "factual",
              ["反向传播", "神经网络", "梯度", "权重"], "medium"),
    TestQuery("F5", "微服务架构的核心特征是什么？", "factual",
              ["微服务", "独立部署", "松耦合"], "medium"),
    TestQuery("F6", "什么是区块链技术中的共识机制？", "factual",
              ["区块链", "共识", "PoW", "PoS"], "hard"),
    TestQuery("F7", "量子计算中的叠加态和纠缠态有什么区别？", "factual",
              ["量子", "叠加态", "纠缠态"], "hard"),
    TestQuery("F8", "什么是图数据库中的属性图模型？", "factual",
              ["图数据库", "属性图", "节点", "关系"], "hard"),
    TestQuery("F9", "编译器优化中的 SSA 形式是什么？", "factual",
              ["SSA", "静态单赋值", "编译器", "优化"], "hard"),

    # === 分析类查询 (Analytical) - 10个 ===
    TestQuery("A1", "分析 Python 在数据科学领域流行的原因", "analytical",
              ["Python", "数据科学", "库", "生态"], "easy"),
    TestQuery("A2", "为什么微服务架构适合大型互联网企业？", "analytical",
              ["微服务", "扩展性", "团队", "部署"], "medium"),
    TestQuery("A3", "分析云原生技术对企业IT架构的影响", "analytical",
              ["云原生", "容器", "Kubernetes", "DevOps"], "medium"),
    TestQuery("A4", "为什么 Rust 语言在系统编程中越来越受欢迎？", "analytical",
              ["Rust", "内存安全", "性能", "并发"], "medium"),
    TestQuery("A5", "分析大语言模型对传统搜索引擎的潜在影响", "analytical",
              ["LLM", "搜索", "AI", "变革"], "hard"),
    TestQuery("A6", "为什么事件驱动架构适合实时数据处理？", "analytical",
              ["事件驱动", "实时", "消息队列", "流处理"], "medium"),
    TestQuery("A7", "分析零信任安全模型在现代企业中的应用价值", "analytical",
              ["零信任", "安全", "身份验证", "网络"], "hard"),
    TestQuery("A8", "为什么函数式编程在并发处理中有优势？", "analytical",
              ["函数式", "并发", "不可变性", "副作用"], "hard"),
    TestQuery("A9", "分析边缘计算对物联网应用的影响", "analytical",
              ["边缘计算", "IoT", "延迟", "带宽"], "medium"),

    # === 对比类查询 (Comparative) - 10个 ===
    TestQuery("C1", "比较 Python 和 JavaScript 在Web开发中的优缺点", "comparative",
              ["Python", "JavaScript", "Web", "对比"], "easy"),
    TestQuery("C2", "对比 SQL 和 NoSQL 数据库的适用场景", "comparative",
              ["SQL", "NoSQL", "数据库", "对比"], "easy"),
    TestQuery("C3", "比较 REST API 和 GraphQL 的设计哲学差异", "comparative",
              ["REST", "GraphQL", "API", "对比"], "medium"),
    TestQuery("C4", "对比 Kubernetes 和 Docker Swarm 的编排能力", "comparative",
              ["Kubernetes", "Docker Swarm", "编排", "对比"], "medium"),
    TestQuery("C5", "比较 React 和 Vue 在前端开发中的特点", "comparative",
              ["React", "Vue", "前端", "对比"], "easy"),
    TestQuery("C6", "对比 Apache Kafka 和 RabbitMQ 的消息处理能力", "comparative",
              ["Kafka", "RabbitMQ", "消息队列", "对比"], "medium"),
    TestQuery("C7", "比较 TensorFlow 和 PyTorch 在深度学习中的特点", "comparative",
              ["TensorFlow", "PyTorch", "深度学习", "对比"], "medium"),
    TestQuery("C8", "对比 gRPC 和传统 HTTP/JSON API 的性能差异", "comparative",
              ["gRPC", "HTTP", "API", "性能"], "hard"),
    TestQuery("C9", "比较 MongoDB 和 PostgreSQL 的数据模型差异", "comparative",
              ["MongoDB", "PostgreSQL", "数据库", "对比"], "medium"),

    # === 时效类查询 (Temporal) - 5个 ===
    TestQuery("T1", "2024年人工智能领域有哪些重要进展？", "temporal",
              ["2024", "AI", "进展", "突破"], "medium"),
    TestQuery("T2", "最近一年云计算技术有哪些新趋势？", "temporal",
              ["云计算", "趋势", "2024", "技术"], "medium"),
    TestQuery("T3", "2024年大语言模型领域有哪些重要发布？", "temporal",
              ["2024", "LLM", "模型", "发布"], "medium"),
    TestQuery("T4", "近期网络安全领域有哪些新的威胁和防护技术？", "temporal",
              ["网络安全", "威胁", "防护", "新"], "medium"),
    TestQuery("T5", "2024年编程语言趋势有哪些变化？", "temporal",
              ["2024", "编程语言", "趋势", "流行"], "easy"),

    # === 技术深度类查询 (Technical) - 5个 ===
    TestQuery("Tech1", "解释 Redis 的内存淘汰策略和持久化机制", "technical",
              ["Redis", "淘汰", "持久化", "内存"], "medium"),
    TestQuery("Tech2", "详解 Kubernetes 中的 Pod 调度算法", "technical",
              ["Kubernetes", "Pod", "调度", "算法"], "hard"),
    TestQuery("Tech3", "解释数据库索引的 B+ 树实现原理", "technical",
              ["索引", "B+树", "数据库", "实现"], "hard"),
    TestQuery("Tech4", "详解 Go 语言的垃圾回收机制", "technical",
              ["Go", "垃圾回收", "GC", "机制"], "hard"),
    TestQuery("Tech5", "解释 Linux 内核的进程调度算法 CFS", "technical",
              ["Linux", "CFS", "调度", "内核"], "hard"),
]


# =============================================================================
# 评估指标定义
# =============================================================================

@dataclass
class TimingMetrics:
    """时间性能指标"""
    total_time: float  # 总执行时间
    planner_time: float  # 规划阶段时间
    search_time: float  # 搜索/检索阶段时间
    fusion_time: float  # 证据融合阶段时间
    writer_time: float  # 报告生成阶段时间

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class QualityMetrics:
    """报告质量指标（通过LLM评估）"""
    relevance_score: float  # 相关性评分 (0-10)
    accuracy_score: float  # 准确性评分 (0-10)
    completeness_score: float  # 完整性评分 (0-10)
    coherence_score: float  # 连贯性评分 (0-10)
    source_citation_score: float  # 来源引用评分 (0-10)
    overall_score: float  # 综合评分 (0-10)

    # 内容分析
    topic_coverage: Dict[str, bool]  # 期望主题覆盖情况
    key_findings_count: int  # 关键发现数量

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EfficiencyMetrics:
    """效率指标"""
    evidence_compression_ratio: float  # 证据压缩率 (deep_rag / fast_web)
    token_efficiency: float  # Token使用效率
    evidence_to_report_ratio: float  # 证据到报告的转化率

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TestResult:
    """单个测试的完整结果"""
    query: TestQuery
    fast_web_result: Dict[str, Any]
    deep_rag_result: Dict[str, Any]
    timing_comparison: Dict[str, Any]
    quality_comparison: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": asdict(self.query),
            "fast_web_result": self.fast_web_result,
            "deep_rag_result": self.deep_rag_result,
            "timing_comparison": self.timing_comparison,
            "quality_comparison": self.quality_comparison,
            "timestamp": self.timestamp,
        }


# =============================================================================
# 评估器实现
# =============================================================================

class ReportEvaluator:
    """报告质量评估器"""

    def __init__(self):
        self.llm = None  # 初始化LLM用于评估

    def evaluate_quality(self, report: str, query: TestQuery) -> QualityMetrics:
        """
        使用LLM评估报告质量

        评估维度：
        1. 相关性：报告内容与查询的相关程度
        2. 准确性：信息的准确性和事实正确性
        3. 完整性：是否覆盖了查询的各个方面
        4. 连贯性：报告结构和逻辑的连贯程度
        5. 来源引用：是否恰当引用信息来源
        """
        # TODO: 实现基于LLM的评估
        # 这里返回占位值
        return QualityMetrics(
            relevance_score=0.0,
            accuracy_score=0.0,
            completeness_score=0.0,
            coherence_score=0.0,
            source_citation_score=0.0,
            overall_score=0.0,
            topic_coverage={},
            key_findings_count=0,
        )

    def compare_reports(
        self,
        fast_report: str,
        deep_report: str,
        query: TestQuery
    ) -> Dict[str, Any]:
        """对比两份报告的质量差异"""
        fast_quality = self.evaluate_quality(fast_report, query)
        deep_quality = self.evaluate_quality(deep_report, query)

        return {
            "fast_web_quality": fast_quality.to_dict(),
            "deep_rag_quality": deep_quality.to_dict(),
            "quality_diff": {
                k: deep_quality.to_dict()[k] - fast_quality.to_dict()[k]
                for k in fast_quality.to_dict().keys()
                if isinstance(fast_quality.to_dict()[k], (int, float))
            },
        }


# =============================================================================
# 基准测试运行器
# =============================================================================

class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        self.evaluator = ReportEvaluator()
        self.results: List[TestResult] = []

        os.makedirs(output_dir, exist_ok=True)

    def _run_single_query(
        self,
        test_query: TestQuery,
        temp_dir: str,
        collection_name: str
    ) -> TestResult:
        """运行单个查询的对比测试"""

        print(f"\n{'='*60}")
        print(f"测试查询 [{test_query.id}]: {test_query.query}")
        print(f"类别: {test_query.category}, 难度: {test_query.difficulty}")
        print(f"{'='*60}")

        # ==========================================================================
        # Phase 1: Fast Web 模式（填充缓存）
        # ==========================================================================
        print("\n>>> Phase 1: Fast Web 模式（收集搜索结果）...")

        fast_state = ResearchState(
            query=test_query.query,
            mode="fast_web",
            hitl_enabled=False,
            test_mode=True  # 启用测试模式，会填充 web_search_cache
        )

        graph = build_graph()

        # 记录时间
        start_time = time.time()
        fast_result_raw = graph.invoke(fast_state)
        fast_total_time = time.time() - start_time

        if isinstance(fast_result_raw, dict):
            fast_result_state = ResearchState(**fast_result_raw)
        else:
            fast_result_state = fast_result_raw

        # 提取缓存
        cached_results = list(fast_result_state.web_search_cache)
        research_tasks = list(fast_result_state.research_tasks)

        print(f"  ✓ 研究任务数: {len(research_tasks)}")
        print(f"  ✓ 缓存的搜索结果: {len(cached_results)}")
        print(f"  ✓ 总执行时间: {fast_total_time:.2f}s")
        print(f"  ✓ 报告长度: {len(fast_result_state.report_draft)} 字符")

        # ==========================================================================
        # Phase 2: Deep RAG 模式（复用缓存）
        # ==========================================================================
        print("\n>>> Phase 2: Deep RAG 模式（复用相同搜索结果）...")

        import agent.retriever as retriever_module
        original_get_retriever = retriever_module.get_retriever

        def get_temp_retriever():
            return retriever_module.Retriever(
                collection_name=collection_name,
                persist_dir=temp_dir
            )
        retriever_module.get_retriever = get_temp_retriever

        try:
            deep_state = ResearchState(
                query=test_query.query,
                mode="deep_rag",
                hitl_enabled=False,
                test_mode=True,
                web_search_cache=cached_results  # 复用相同的缓存
            )

            start_time = time.time()
            deep_result_raw = graph.invoke(deep_state)
            deep_total_time = time.time() - start_time

            if isinstance(deep_result_raw, dict):
                deep_result_state = ResearchState(**deep_result_raw)
            else:
                deep_result_state = deep_result_raw

            print(f"  ✓ 检索到的证据数: {len(deep_result_state.retrieved_evidence)}")
            print(f"  ✓ 总执行时间: {deep_total_time:.2f}s")
            print(f"  ✓ 报告长度: {len(deep_result_state.report_draft)} 字符")

        finally:
            retriever_module.get_retriever = original_get_retriever

        # ==========================================================================
        # Phase 3: 收集指标和对比
        # ==========================================================================
        print("\n>>> Phase 3: 对比分析...")

        # 时间对比
        timing_comparison = {
            "fast_web_total": fast_total_time,
            "deep_rag_total": deep_total_time,
            "time_diff": deep_total_time - fast_total_time,
            "time_diff_percent": ((deep_total_time - fast_total_time) / fast_total_time * 100)
                                if fast_total_time > 0 else 0,
            "winner": "fast_web" if fast_total_time < deep_total_time else "deep_rag"
        }

        # 证据数量对比
        evidence_comparison = {
            "fast_web_evidence_count": len(fast_result_state.retrieved_evidence),
            "deep_rag_evidence_count": len(deep_result_state.retrieved_evidence),
            "compression_ratio": len(deep_result_state.retrieved_evidence) / len(fast_result_state.retrieved_evidence)
                                         if len(fast_result_state.retrieved_evidence) > 0 else 1.0
        }

        # 报告长度对比
        report_comparison = {
            "fast_web_report_length": len(fast_result_state.report_draft),
            "deep_rag_report_length": len(deep_result_state.report_draft),
            "length_diff": len(deep_result_state.report_draft) - len(fast_result_state.report_draft)
        }

        print(f"  时间对比:")
        print(f"    Fast Web: {fast_total_time:.2f}s")
        print(f"    Deep RAG: {deep_total_time:.2f}s")
        print(f"    差异: {timing_comparison['time_diff']:.2f}s ({timing_comparison['time_diff_percent']:.1f}%)")
        print(f"    胜出: {timing_comparison['winner']}")

        print(f"  证据数量对比:")
        print(f"    Fast Web: {evidence_comparison['fast_web_evidence_count']}")
        print(f"    Deep RAG: {evidence_comparison['deep_rag_evidence_count']}")
        print(f"    压缩率: {evidence_comparison['compression_ratio']:.2f}")

        # 构建测试结果
        test_result = TestResult(
            query=test_query,
            fast_web_result={
                "research_tasks": research_tasks,
                "evidence_count": len(fast_result_state.retrieved_evidence),
                "report": fast_result_state.report_draft,
                "execution_time": fast_total_time,
            },
            deep_rag_result={
                "evidence_count": len(deep_result_state.retrieved_evidence),
                "report": deep_result_state.report_draft,
                "execution_time": deep_total_time,
            },
            timing_comparison=timing_comparison,
            quality_comparison={
                "evidence": evidence_comparison,
                "report": report_comparison,
            },
            timestamp=datetime.now().isoformat(),
        )

        return test_result

    def run_benchmark(
        self,
        queries: Optional[List[TestQuery]] = None,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        运行完整基准测试

        Args:
            queries: 要测试的查询列表，默认使用 BENCHMARK_QUERIES
            parallel: 是否并行运行（暂不支持）

        Returns:
            完整的测试结果汇总
        """
        queries = queries or BENCHMARK_QUERIES

        print(f"\n{'='*80}")
        print(f"启动基准测试套件")
        print(f"测试查询数量: {len(queries)}")
        print(f"测试时间: {datetime.now().isoformat()}")
        print(f"{'='*80}\n")

        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="benchmark_")

        results = []
        errors = []

        for i, query in enumerate(queries, 1):
            print(f"\n{'#'*80}")
            print(f"进度: {i}/{len(queries)} ({i/len(queries)*100:.1f}%)")
            print(f"{'#'*80}")

            try:
                result = self._run_single_query(
                    query,
                    temp_dir,
                    f"benchmark_{query.id}_{int(time.time())}"
                )
                results.append(result)

                # 每完成一个测试就保存结果
                self._save_intermediate_results(results, temp_dir)

            except Exception as e:
                print(f"\n[错误] 测试 {query.id} 失败: {str(e)}")
                errors.append({
                    "query_id": query.id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

        # 生成汇总报告
        summary = self._generate_summary(results, errors)

        # 保存最终结果
        final_report_path = self._save_final_report(summary, results, errors)

        print(f"\n{'='*80}")
        print(f"基准测试完成!")
        print(f"成功: {len(results)}/{len(queries)}")
        print(f"失败: {len(errors)}/{len(queries)}")
        print(f"报告保存至: {final_report_path}")
        print(f"{'='*80}\n")

        return summary

    def _save_intermediate_results(self, results: List[TestResult], temp_dir: str):
        """保存中间结果"""
        intermediate_path = os.path.join(temp_dir, "intermediate_results.json")
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    def _generate_summary(
        self,
        results: List[TestResult],
        errors: List[Dict]
    ) -> Dict[str, Any]:
        """生成测试汇总"""

        if not results:
            return {
                "status": "failed",
                "message": "没有成功的测试结果",
                "errors": errors,
            }

        # 时间统计
        fast_times = [r.timing_comparison["fast_web_total"] for r in results]
        deep_times = [r.timing_comparison["deep_rag_total"] for r in results]

        time_diffs = [r.timing_comparison["time_diff"] for r in results]

        timing_stats = {
            "fast_web": {
                "mean": statistics.mean(fast_times),
                "median": statistics.median(fast_times),
                "min": min(fast_times),
                "max": max(fast_times),
                "stdev": statistics.stdev(fast_times) if len(fast_times) > 1 else 0,
            },
            "deep_rag": {
                "mean": statistics.mean(deep_times),
                "median": statistics.median(deep_times),
                "min": min(deep_times),
                "max": max(deep_times),
                "stdev": statistics.stdev(deep_times) if len(deep_times) > 1 else 0,
            },
            "time_diff": {
                "mean": statistics.mean(time_diffs),
                "median": statistics.median(time_diffs),
                "fast_wins": sum(1 for d in time_diffs if d > 0),
                "deep_wins": sum(1 for d in time_diffs if d < 0),
            },
        }

        # 证据压缩统计
        compression_ratios = [
            r.quality_comparison["evidence"]["compression_ratio"]
            for r in results
        ]

        evidence_stats = {
            "compression_ratio": {
                "mean": statistics.mean(compression_ratios),
                "median": statistics.median(compression_ratios),
                "min": min(compression_ratios),
                "max": max(compression_ratios),
            },
        }

        # 按类别统计
        category_stats = {}
        for category in ["factual", "analytical", "comparative", "temporal", "technical"]:
            cat_results = [r for r in results if r.query.category == category]
            if cat_results:
                cat_fast_times = [r.timing_comparison["fast_web_total"] for r in cat_results]
                cat_deep_times = [r.timing_comparison["deep_rag_total"] for r in cat_results]

                category_stats[category] = {
                    "count": len(cat_results),
                    "fast_web_mean_time": statistics.mean(cat_fast_times),
                    "deep_rag_mean_time": statistics.mean(cat_deep_times),
                }

        return {
            "status": "success",
            "summary": {
                "total_queries": len(results) + len(errors),
                "successful": len(results),
                "failed": len(errors),
                "success_rate": len(results) / (len(results) + len(errors)) * 100,
            },
            "timing_statistics": timing_stats,
            "evidence_statistics": evidence_stats,
            "category_statistics": category_stats,
            "errors": errors,
        }

    def _save_final_report(
        self,
        summary: Dict,
        results: List[TestResult],
        errors: List[Dict]
    ) -> str:
        """保存最终报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.output_dir,
            f"benchmark_report_{timestamp}.json"
        )

        full_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(results) + len(errors),
                "output_dir": self.output_dir,
            },
            "summary": summary,
            "detailed_results": [r.to_dict() for r in results],
            "errors": errors,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)

        # 同时生成一个简化的Markdown报告
        md_report_path = report_path.replace(".json", ".md")
        self._generate_markdown_report(full_report, md_report_path)

        return report_path

    def _generate_markdown_report(self, full_report: Dict, output_path: str):
        """生成Markdown格式的报告"""
        summary = full_report["summary"]
        timing = summary.get("timing_statistics", {})

        md_content = f"""# Fast Web vs Deep RAG 基准测试报告

## 测试概况

- **测试时间**: {full_report['metadata']['timestamp']}
- **总查询数**: {summary['summary']['total_queries']}
- **成功**: {summary['summary']['successful']}
- **失败**: {summary['summary']['failed']}
- **成功率**: {summary['summary']['success_rate']:.1f}%

## 时间性能对比

| 指标 | Fast Web | Deep RAG | 差异 |
|------|----------|----------|------|
| 平均执行时间 (s) | {timing.get('fast_web', {}).get('mean', 0):.2f} | {timing.get('deep_rag', {}).get('mean', 0):.2f} | {timing.get('time_diff', {}).get('mean', 0):.2f} |
| 中位数 (s) | {timing.get('fast_web', {}).get('median', 0):.2f} | {timing.get('deep_rag', {}).get('median', 0):.2f} | - |
| 最小值 (s) | {timing.get('fast_web', {}).get('min', 0):.2f} | {timing.get('deep_rag', {}).get('min', 0):.2f} | - |
| 最大值 (s) | {timing.get('fast_web', {}).get('max', 0):.2f} | {timing.get('deep_rag', {}).get('max', 0):.2f} | - |

## 类别统计

"""

        # 添加类别统计
        cat_stats = summary.get("category_statistics", {})
        for cat, stats in cat_stats.items():
            md_content += f"\n### {cat}\n"
            md_content += f"- 查询数量: {stats['count']}\n"
            md_content += f"- Fast Web 平均时间: {stats['fast_web_mean_time']:.2f}s\n"
            md_content += f"- Deep RAG 平均时间: {stats['deep_rag_mean_time']:.2f}s\n"

        md_content += "\n## 详细结果\n\n"

        # 添加每个查询的详细结果链接
        for i, result in enumerate(full_report.get("detailed_results", [])):
            query = result["query"]
            md_content += f"\n### {query['id']}: {query['query']}\n"
            md_content += f"- 类别: {query['category']}\n"
            md_content += f"- Fast Web 时间: {result['timing_comparison']['fast_web_total']:.2f}s\n"
            md_content += f"- Deep RAG 时间: {result['timing_comparison']['deep_rag_total']:.2f}s\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)


# =============================================================================
# 测试函数
# =============================================================================

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_small_benchmark():
    """
    小规模基准测试（5个查询）

    用于快速验证测试框架和对比效果。
    """
    runner = BenchmarkRunner(output_dir="./benchmark_results_small")

    # 选择5个不同类型的查询
    selected_queries = [
        BENCHMARK_QUERIES[0],   # F1 - factual easy
        BENCHMARK_QUERIES[10],  # A1 - analytical easy
        BENCHMARK_QUERIES[20],  # C1 - comparative easy
        BENCHMARK_QUERIES[45],  # T1 - temporal
        BENCHMARK_QUERIES[50], # Tech1 - technical
    ]

    summary = runner.run_benchmark(selected_queries)

    # 验证结果
    assert summary["status"] == "success", f"基准测试失败: {summary.get('message')}"
    assert summary["summary"]["success_rate"] >= 80, "成功率低于80%"

    return summary


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_medium_benchmark():
    """
    中等规模基准测试（20个查询，每类4个）

    用于获得统计上更有意义的对比结果。
    """
    runner = BenchmarkRunner(output_dir="./benchmark_results_medium")

    # 选择20个查询，每类4个
    selected_queries = []
    categories = ["factual", "analytical", "comparative", "temporal", "technical"]

    for cat in categories:
        cat_queries = [q for q in BENCHMARK_QUERIES if q.category == cat]
        selected_queries.extend(cat_queries[:4])

    summary = runner.run_benchmark(selected_queries)

    assert summary["status"] == "success"
    assert summary["summary"]["success_rate"] >= 80

    return summary


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run LLM tests"
)
@pytest.mark.e2e
def test_full_benchmark():
    """
    完整基准测试（全部50个查询）

    用于发布正式对比报告。运行时间较长（预计1-2小时）。
    """
    runner = BenchmarkRunner(output_dir="./benchmark_results_full")

    summary = runner.run_benchmark(BENCHMARK_QUERIES)

    assert summary["status"] == "success"

    return summary


# =============================================================================
# 辅助函数和CLI入口
# =============================================================================

def run_quick_comparison(query: str, output_dir: str = "./quick_comparison"):
    """
    快速对比单个查询

    用于快速验证或调试特定查询的对比效果。

    Usage:
        python -c "from tests.test_benchmark_suite import run_quick_comparison; run_quick_comparison('你的查询')"
    """
    runner = BenchmarkRunner(output_dir=output_dir)

    test_query = TestQuery(
        id="QUICK",
        query=query,
        category="custom",
        expected_topics=[],
        difficulty="unknown"
    )

    temp_dir = tempfile.mkdtemp()
    collection_name = f"quick_{int(time.time())}"

    try:
        result = runner._run_single_query(test_query, temp_dir, collection_name)

        print("\n" + "="*80)
        print("快速对比结果")
        print("="*80)
        print(f"\n查询: {query}")
        print(f"\n时间对比:")
        print(f"  Fast Web: {result.timing_comparison['fast_web_total']:.2f}s")
        print(f"  Deep RAG: {result.timing_comparison['deep_rag_total']:.2f}s")
        print(f"  差异: {result.timing_comparison['time_diff']:.2f}s")
        print(f"\n报告长度:")
        print(f"  Fast Web: {result.quality_comparison['report']['fast_web_report_length']} 字符")
        print(f"  Deep RAG: {result.quality_comparison['report']['deep_rag_report_length']} 字符")

        # 保存结果
        result_file = os.path.join(output_dir, f"quick_result_{int(time.time())}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"\n详细结果已保存: {result_file}")

        return result

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # 允许命令行运行快速对比
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"运行快速对比: {query}")
        run_quick_comparison(query)
    else:
        # 运行 pytest
        pytest.main([__file__, "-v", "-s", "-m", "e2e"])
