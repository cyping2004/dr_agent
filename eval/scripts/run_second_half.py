#!/usr/bin/env python3
"""
执行后半段脚本（串行执行）

用法:
    python -m eval.scripts.run_second_half \
        --cache-dir eval/cache/basic_10 \
        --output-dir eval/results/run_20240101_120000 \
        --top-k 5

功能:
    1. 加载前半段缓存数据
    2. 对每个查询串行执行两种模式的后半段：
       - Fast Web: documents → Writer
       - Deep RAG: documents → Ingest → Retrieve → Writer
    3. 收集和记录指标
    4. 生成对比报告
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graph.split_graph import (
    SplitResearchGraph,
    FirstHalfOutput,
    SecondHalfResult,
    SecondHalfMetrics
)
from eval.utils.cache_manager import CacheManager
from eval.utils.metrics_recorder import MetricsRecorder


def format_duration(ms: float) -> str:
    """格式化持续时间"""
    if ms < 1000:
        return f"{ms:.0f}ms"
    else:
        return f"{ms/1000:.2f}s"


def run_second_half(
    cache_dir: str,
    output_dir: str,
    top_k: int = 5,
    collection_prefix: str = "test_deep_rag",
    score_threshold: float | None = None
) -> None:
    """
    执行后半段主函数（串行执行）

    Args:
        cache_dir: 前半段缓存目录
        output_dir: 输出目录
        top_k: Deep RAG检索Top-K
        collection_prefix: 向量集合名称前缀
    """
    print("=" * 70)
    print("执行后半段对比测试（串行执行）")
    print("=" * 70)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reports_dir = output_path / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 加载缓存
    print(f"\n[1/5] 加载前半段缓存")
    print(f"      缓存目录: {cache_dir}")

    cache_manager = CacheManager(cache_dir=cache_dir)
    cached_outputs = cache_manager.load_all_cached_queries_with_ids()

    if not cached_outputs:
        print("[错误] 没有找到缓存数据，请先运行前半段脚本")
        return

    print(f"      加载了 {len(cached_outputs)} 个查询")

    # 初始化组件
    print(f"\n[2/5] 初始化组件")
    print(f"      Top-K: {top_k}")
    if score_threshold is None:
        print("      相似度阈值: 未启用")
    else:
        print(f"      相似度阈值: {score_threshold}")
    print(f"      执行模式: 串行执行（先Fast Web，后Deep RAG）")

    graph = SplitResearchGraph(top_k=top_k, collection_prefix=collection_prefix)
    recorder = MetricsRecorder(output_dir=str(output_path))

    # 执行对比测试
    print(f"\n[3/5] 执行对比测试")
    print("-" * 70)

    for i, (query_id, first_half_output) in enumerate(cached_outputs):
        query = first_half_output.query
        if not query_id:
            query_id = f"query_{i+1:03d}"

        print(f"\n[{i+1}/{len(cached_outputs)}] 查询: {query[:50]}...")
        print(f"       查询ID: {query_id}")

        original_doc_count = len(first_half_output.documents)
        original_tokens = sum(
            len(doc.page_content.split())
            for doc in first_half_output.documents
        )

        fast_metrics = {}
        deep_metrics = {}

        # ========== 串行执行Fast Web ==========
        print(f"       [Fast Web] 执行中...")
        try:
            fast_result = graph.run_fast_web_second_half(first_half_output)

            # 记录指标
            fast_metrics = {
                "total_backend_time": fast_result.metrics.total_time_ms,
                "writer_time": fast_result.metrics.writer_time_ms,
                "original_doc_count": original_doc_count,
                "retrieved_doc_count": fast_result.metrics.retrieved_doc_count,
                "original_tokens": original_tokens,
                "retrieved_tokens": fast_result.metrics.retrieved_tokens,
                "input_tokens": fast_result.metrics.input_tokens,
                "output_tokens": fast_result.metrics.output_tokens
            }

            # 保存报告
            report_path = reports_dir / f"{query_id}_fast_web.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Fast Web 报告\n\n")
                f.write(f"**查询**: {query}\n\n")
                f.write(f"**模式**: Fast Web\n\n")
                f.write(f"**执行时间**: {fast_result.metrics.total_time_ms:.2f}ms\n\n")
                f.write(f"---\n\n")
                f.write(fast_result.report)

            print(f"       [Fast Web] ✓ 成功 ({fast_result.metrics.total_time_ms:.0f}ms)")

        except Exception as e:
            print(f"       [Fast Web] ✗ 失败: {e}")
            import traceback
            traceback.print_exc()

        # ========== 串行执行Deep RAG ==========
        print(f"       [Deep RAG] 执行中...")
        try:
            deep_result = graph.run_deep_rag_second_half(
                first_half_output,
                top_k=top_k,
                score_threshold=score_threshold
            )

            compression_ratio = 0.0
            compression_ratio_tokens = 0.0
            if deep_result.metrics.retrieved_chunk_count > 0:
                compression_ratio = (
                    deep_result.metrics.original_chunk_count
                    / deep_result.metrics.retrieved_chunk_count
                )
            if deep_result.metrics.retrieved_tokens > 0:
                compression_ratio_tokens = (
                    deep_result.metrics.original_tokens
                    / deep_result.metrics.retrieved_tokens
                )

            # 记录指标
            deep_metrics = {
                "total_backend_time": deep_result.metrics.total_time_ms,
                "ingest_time": deep_result.metrics.ingest_time_ms,
                "retrieve_time": deep_result.metrics.retrieve_time_ms,
                "writer_time": deep_result.metrics.writer_time_ms,
                "original_doc_count": original_doc_count,
                "retrieved_doc_count": deep_result.metrics.retrieved_doc_count,
                "original_chunk_count": deep_result.metrics.original_chunk_count,
                "retrieved_chunk_count": deep_result.metrics.retrieved_chunk_count,
                "original_tokens": deep_result.metrics.original_tokens,
                "retrieved_tokens": deep_result.metrics.retrieved_tokens,
                "input_tokens": deep_result.metrics.input_tokens,
                "output_tokens": deep_result.metrics.output_tokens,
                "compression_ratio": compression_ratio,
                "compression_ratio_tokens": compression_ratio_tokens
            }

            # 保存报告
            report_path = reports_dir / f"{query_id}_deep_rag.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Deep RAG 报告\n\n")
                f.write(f"**查询**: {query}\n\n")
                f.write(f"**模式**: Deep RAG (Top-K={top_k})\n\n")
                if score_threshold is not None:
                    f.write(f"**相似度阈值**: {score_threshold}\n\n")
                f.write(f"**执行时间**: {deep_result.metrics.total_time_ms:.2f}ms\n\n")
                f.write(f"**压缩比**: {compression_ratio:.2f}x\n\n")
                f.write(f"**Token压缩比**: {compression_ratio_tokens:.2f}x\n\n")
                f.write(f"---\n\n")
                f.write(deep_result.report)

            print(f"       [Deep RAG] ✓ 成功 ({deep_result.metrics.total_time_ms:.0f}ms, 压缩比: {compression_ratio:.2f}x)")

        except Exception as e:
            print(f"       [Deep RAG] ✗ 失败: {e}")
            import traceback
            traceback.print_exc()

        if fast_metrics or deep_metrics:
            fast_total = fast_metrics.get("total_backend_time", 0)
            rag_total = deep_metrics.get("total_backend_time", 0)
            timing_comparison = {
                "fast_web_total_ms": fast_total,
                "deep_rag_total_ms": rag_total,
                "time_diff_ms": rag_total - fast_total
            }
            if fast_total > 0:
                timing_comparison["time_overhead_percent"] = ((rag_total - fast_total) / fast_total) * 100
            winner = "fast_web" if (fast_total and (fast_total <= rag_total or rag_total == 0)) else "deep_rag"

            recorder.record_comparison_result(
                query_id=query_id,
                query=query,
                fast_web_metrics=fast_metrics,
                deep_rag_metrics=deep_metrics,
                timing_comparison=timing_comparison,
                winner=winner
            )

    # 完成
    print("-" * 70)
    print(f"\n[4/5] 保存结果")

    # 导出指标
    csv_path = output_path / "metrics.csv"
    recorder.export_to_csv(str(csv_path))

    json_path = output_path / "detailed_metrics.json"
    recorder.export_to_json(str(json_path))

    # 生成摘要报告
    summary_path = output_path / "summary.md"
    summary = recorder.generate_summary_report()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 对比测试摘要\n\n")
        f.write(f"**测试时间**: {datetime.now().isoformat()}\n\n")
        f.write(f"**总查询数**: {summary.get('total_queries', 0)}\n\n")

        if "error" not in summary:
            f.write("## 性能指标\n\n")
            f.write(f"- Fast Web 平均时间: {summary['fast_web']['avg_total_time_ms']:.2f}ms\n")
            f.write(f"- Deep RAG 平均时间: {summary['deep_rag']['avg_total_time_ms']:.2f}ms\n")
            f.write(f"- 平均时间开销: {summary['comparison']['time_overhead_percent']:.2f}%\n")
            f.write(f"- 平均压缩比: {summary['compression']['avg_compression_ratio']:.2f}x\n")
            f.write(f"- 平均Token压缩比: {summary['compression']['avg_compression_ratio_tokens']:.2f}x\n")
        else:
            f.write(f"- 摘要生成失败: {summary['error']}\n")

    print(f"      指标文件: {csv_path}")
    print(f"      详细指标: {json_path}")
    print(f"      摘要报告: {summary_path}")
    print(f"      报告目录: {reports_dir}")

    print(f"\n[5/5] 摘要")
    if "error" not in summary:
        print(f"      Fast Web 平均时间: {summary['fast_web']['avg_total_time_ms']:.2f}ms")
        print(f"      Deep RAG 平均时间: {summary['deep_rag']['avg_total_time_ms']:.2f}ms")
        print(f"      平均压缩比: {summary['compression']['avg_compression_ratio']:.2f}x")
        print(f"      平均Token压缩比: {summary['compression']['avg_compression_ratio_tokens']:.2f}x")
    print(f"\n全部完成！")
    print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="执行后半段对比测试（串行执行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认参数
    python -m eval.scripts.run_second_half

    # 指定缓存目录和输出目录
    python -m eval.scripts.run_second_half \\
        --cache-dir eval/cache/basic_10 \\
        --output-dir eval/results/run_20240101_120000

    # 指定Top-K值
    python -m eval.scripts.run_second_half \\
        --top-k 10
"""
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="eval/cache",
        help="前半段缓存目录 (默认: eval/cache)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: eval/results/run_YYYYMMDD_HHMMSS)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Deep RAG检索Top-K (默认: 5)"
    )

    parser.add_argument(
        "--collection-prefix",
        type=str,
        default="test_deep_rag",
        help="向量集合名称前缀 (默认: test_deep_rag)"
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="相似度阈值（dense模式为最大距离，其它模式为最小分数）"
    )

    args = parser.parse_args()

    # 如果未指定输出目录，自动生成
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"eval/results/run_{timestamp}"

    run_second_half(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        collection_prefix=args.collection_prefix,
        score_threshold=args.score_threshold
    )


if __name__ == "__main__":
    main()
