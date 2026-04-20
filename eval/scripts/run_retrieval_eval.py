#!/usr/bin/env python3
"""
Retrieval-only evaluation for Deep RAG.

This script runs ingestion + retrieval only (no report generation), then computes
Recall@k, Hit@k, and MRR against gold URLs.

Usage:
    python -m eval.scripts.run_retrieval_eval \
        --dataset eval/datasets/retrieval_web_template.json \
        --cache-dir eval/cache/basic_10 \
        --output-dir eval/results/retrieval_YYYYMMDD_HHMMSS \
        --k-list 1 3 5
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Iterable

from eval.utils.cache_manager import CacheManager
from graph.split_graph import SplitResearchGraph


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def _dedup_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _extract_urls(docs) -> List[str]:
    urls = []
    for doc in docs:
        meta = doc.metadata or {}
        url = meta.get("url") or meta.get("source_url")
        if not url:
            continue
        url_str = str(url)
        if url_str.startswith("http://") or url_str.startswith("https://"):
            urls.append(url_str)
    return urls


def _compute_metrics_for_k(retrieved_urls: List[str], gold_urls: List[str], k: int) -> Dict[str, float]:
    if k <= 0:
        return {"recall": 0.0, "hit": 0.0, "mrr": 0.0}

    gold_set = set(gold_urls)
    if not gold_set:
        return {"recall": 0.0, "hit": 0.0, "mrr": 0.0}

    top_urls = retrieved_urls[:k]
    hits = [url for url in top_urls if url in gold_set]

    recall = len(set(hits)) / len(gold_set)
    hit = 1.0 if hits else 0.0

    mrr = 0.0
    for idx, url in enumerate(top_urls, start=1):
        if url in gold_set:
            mrr = 1.0 / float(idx)
            break

    return {"recall": recall, "hit": hit, "mrr": mrr}


def _aggregate_metrics(rows: List[Dict[str, Any]], k_list: List[int]) -> Dict[str, Any]:
    summary = {"by_k": {}, "total_queries": 0}
    if not rows:
        return summary

    summary["total_queries"] = len({row["query_id"] for row in rows})
    for k in k_list:
        k_rows = [row for row in rows if row["k"] == k]
        if not k_rows:
            continue
        summary["by_k"][str(k)] = {
            "avg_recall": sum(row["recall"] for row in k_rows) / len(k_rows),
            "avg_hit": sum(row["hit"] for row in k_rows) / len(k_rows),
            "avg_mrr": sum(row["mrr"] for row in k_rows) / len(k_rows),
        }

    return summary


def run_retrieval_eval(
    dataset_path: str,
    cache_dir: str,
    output_dir: str,
    k_list: List[int],
    retrieval_mode: str | None = None,
    score_threshold: float | None = None,
    chunk_size: int = 1024,
    overlap: int = 32,
    collection_prefix: str = "retrieval_eval",
    multimodal: bool = True,
    mm_view: str = "fused",
    reuse_collection: bool = False
) -> None:
    if retrieval_mode:
        os.environ["RETRIEVAL_MODE"] = retrieval_mode

    queries = _load_dataset(dataset_path)
    cache_manager = CacheManager(cache_dir=cache_dir)
    cached = dict(cache_manager.load_all_cached_queries_with_ids())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    graph = SplitResearchGraph(top_k=max(k_list), collection_prefix=collection_prefix)

    for entry in queries:
        query_id = entry.get("id")
        query = entry.get("query", "")
        gold_urls = entry.get("gold_urls", [])

        if not query_id or query_id not in cached:
            print(f"[Skip] Missing cache for {query_id}")
            continue

        if not gold_urls:
            print(f"[Skip] Missing gold_urls for {query_id}")
            continue

        first_half_output = cached[query_id]

        result = graph.run_deep_rag_retrieval_only(
            first_half_output,
            top_k=max(k_list),
            score_threshold=score_threshold,
            chunk_size=chunk_size,
            overlap=overlap,
            multimodal=multimodal,
            mm_view=mm_view,
            reuse_collection=reuse_collection,
        )

        retrieved_urls = _dedup_preserve(_extract_urls(result.retrieved_docs))
        gold_set = set(gold_urls)

        for k in k_list:
            metrics = _compute_metrics_for_k(retrieved_urls, gold_urls, k)
            rows.append({
                "query_id": query_id,
                "query": query,
                "k": k,
                "recall": metrics["recall"],
                "hit": metrics["hit"],
                "mrr": metrics["mrr"],
                "gold_count": len(gold_set),
                "retrieved_count": len(retrieved_urls),
                "retrieval_mode": os.environ.get("RETRIEVAL_MODE", ""),
                "mm_view": mm_view,
            })

        details.append({
            "query_id": query_id,
            "query": query,
            "gold_urls": gold_urls,
            "retrieved_urls": retrieved_urls,
            "metrics": {
                str(k): _compute_metrics_for_k(retrieved_urls, gold_urls, k)
                for k in k_list
            }
        })

    summary = _aggregate_metrics(rows, k_list)

    csv_path = output_path / "retrieval_metrics.csv"
    json_path = output_path / "retrieval_metrics.json"

    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query",
                "k",
                "recall",
                "hit",
                "mrr",
                "gold_count",
                "retrieved_count",
                "retrieval_mode",
                "mm_view",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_path,
                "cache_dir": cache_dir,
                "k_list": k_list,
                "summary": summary,
                "details": details,
                "mm_view": mm_view,
                "reuse_collection": reuse_collection,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Retrieval Eval] CSV: {csv_path}")
    print(f"[Retrieval Eval] JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only evaluation for Deep RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="eval/cache")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--retrieval-mode", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--collection-prefix", type=str, default="retrieval_eval")
    parser.add_argument(
        "--reuse-collection",
        action="store_true",
        help="复用已构建的向量集合，避免重复embedding",
    )
    parser.add_argument(
        "--no-multimodal",
        action="store_true",
        help="禁用多模态检索（默认开启）",
    )
    parser.add_argument(
        "--mm-view",
        type=str,
        default="fused",
        choices=["fused", "text", "caption", "image"],
        help="多模态检索视图 (默认: fused)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval/results/retrieval_{timestamp}"

    run_retrieval_eval(
        dataset_path=args.dataset,
        cache_dir=args.cache_dir,
        output_dir=output_dir,
        k_list=args.k_list,
        retrieval_mode=args.retrieval_mode,
        score_threshold=args.score_threshold,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        collection_prefix=args.collection_prefix,
        multimodal=not args.no_multimodal,
        mm_view=args.mm_view,
        reuse_collection=args.reuse_collection,
    )


if __name__ == "__main__":
    main()
