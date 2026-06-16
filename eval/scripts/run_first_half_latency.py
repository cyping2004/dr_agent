#!/usr/bin/env python3
"""
Run first-half pipeline (Planner -> WebSearcher) multiple times and report latency.

Usage:
    python -m eval.scripts.run_first_half_latency \
        --dataset eval/datasets/basic_10.json \
        --provider tavily \
        --runs 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graph.split_graph import SplitResearchGraph


def load_dataset(dataset_path: str) -> List[dict]:
    """Load dataset JSON and return query list."""
    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("queries", [])


def run_once(
    run_index: int,
    total_runs: int,
    queries: List[dict],
    graph: SplitResearchGraph
) -> List[float]:
    """Run one pass over all queries and return per-query latencies (ms)."""
    durations: List[float] = []

    print(f"\n=== Run {run_index}/{total_runs} ===")
    for idx, query_data in enumerate(queries, 1):
        query_id = query_data.get("id", f"q{idx:03d}")
        query = query_data.get("query", "")

        start_time = time.perf_counter()
        graph.run_first_half(query=query)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        durations.append(elapsed_ms)
        print(f"[Run {run_index}] {query_id}: {elapsed_ms:.2f} ms")

    return durations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run first-half benchmark multiple times and report average latency."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/datasets/basic_10.json",
        help="Path to dataset JSON (default: eval/datasets/basic_10.json)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="tavily",
        choices=["tavily", "duckduckgo"],
        help="Web search provider (default: tavily)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs (default: 1)",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        nargs="+",
        help="Optional list of query IDs to run",
    )

    args = parser.parse_args()

    if args.runs <= 0:
        print("[Error] --runs must be a positive integer")
        return 1

    if args.provider == "tavily" and not os.getenv("TAVILY_API_KEY"):
        print("[Error] TAVILY_API_KEY not set")
        return 1

    os.environ["WEB_SEARCH_PROVIDER"] = args.provider

    queries = load_dataset(args.dataset)
    if args.query_ids:
        queries = [q for q in queries if q.get("id") in args.query_ids]

    if not queries:
        print("[Error] No queries found to run")
        return 1

    print("=" * 70)
    print("First-half latency benchmark")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Provider: {args.provider}")
    print(f"Runs: {args.runs}")
    print(f"Queries per run: {len(queries)}")

    graph = SplitResearchGraph()

    all_durations: List[float] = []
    run_averages: List[float] = []

    for run_index in range(1, args.runs + 1):
        durations = run_once(run_index, args.runs, queries, graph)
        if durations:
            run_avg = mean(durations)
            run_averages.append(run_avg)
            all_durations.extend(durations)
            print(f"[Run {run_index}] average: {run_avg:.2f} ms")

    if not all_durations:
        print("[Error] No timing data collected")
        return 1

    overall_avg = mean(all_durations)
    min_ms = min(all_durations)
    max_ms = max(all_durations)
    avg_per_run = mean(run_averages) if run_averages else 0.0

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total calls: {len(all_durations)}")
    print(f"Average latency: {overall_avg:.2f} ms")
    print(f"Min/Max latency: {min_ms:.2f} ms / {max_ms:.2f} ms")
    print(f"Average per run: {avg_per_run:.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
