#!/usr/bin/env python3
"""
执行前半段测试脚本

运行前半段（Query → Planner → WebSearcher）并保存缓存数据。

Usage:
    python -m eval.scripts.run_first_half \
        --dataset eval/datasets/basic_10.json \
        --output-dir eval/cache/basic_10 \
        --provider tavily
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graph.split_graph import SplitResearchGraph, FirstHalfOutput
from eval.utils.cache_manager import CacheManager


def load_dataset(dataset_path: str) -> List[dict]:
    """Load test dataset."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('queries', [])


def run_first_half(
    query: str,
    query_id: str,
    graph: SplitResearchGraph
) -> FirstHalfOutput:
    """Execute first half for a single query."""
    print(f"\n{'='*60}")
    print(f"[Query {query_id}] {query}")
    print(f"{'='*60}")

    output = graph.run_first_half(query=query)

    print(f"\n[Summary]")
    print(f"  - Tasks: {len(output.tasks)}")
    print(f"  - Documents: {len(output.documents)}")
    print(f"  - Timestamp: {output.timestamp}")

    return output


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run first half test (Query → Planner → WebSearcher)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset', type=str, default='eval/datasets/basic_10.json')
    parser.add_argument('--output-dir', type=str, default='eval/cache/basic_10')
    parser.add_argument('--provider', type=str, default='tavily', choices=['tavily', 'duckduckgo'])
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--query-ids', type=str, nargs='+')

    args = parser.parse_args()

    # Check environment
    if args.provider == 'tavily' and not os.getenv('TAVILY_API_KEY'):
        print("[Error] TAVILY_API_KEY not set")
        sys.exit(1)

    os.environ['WEB_SEARCH_PROVIDER'] = args.provider

    # Load dataset
    queries = load_dataset(args.dataset)
    if args.query_ids:
        queries = [q for q in queries if q.get('id') in args.query_ids]

    print(f"Loaded {len(queries)} queries")

    # Initialize components
    graph = SplitResearchGraph()
    cache_manager = CacheManager(cache_dir=args.output_dir)

    stats = {"success": 0, "skipped": 0, "failed": 0}
    total_queries = len(queries)

    # Process queries
    for i, query_data in enumerate(queries, 1):
        query_id = query_data.get('id', f'q{i:03d}')
        query = query_data.get('query', '')

        if args.skip_existing and cache_manager.check_cache_exists(query_id):
            print(f"[{i}/{len(queries)}] {query_id} cached, skipping")
            stats["skipped"] += 1
            continue

        try:
            output = run_first_half(query, query_id, graph)
            cache_manager.save_first_half_output(query_id=query_id, output=output)
            stats["success"] += 1
        except Exception as e:
            print(f"[Error] {query_id} failed: {e}")
            import traceback
            traceback.print_exc()
            stats["failed"] += 1

    # Save metadata
    cache_manager.save_metadata({
        "dataset": args.dataset,
        "provider": args.provider,
        "total_queries": total_queries,
        "successful": stats["success"],
        "skipped": stats["skipped"],
        "failed": stats["failed"]
    })

    print(f"\nCompleted: {stats['success']}/{len(queries)} success, {stats['failed']} failed")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
