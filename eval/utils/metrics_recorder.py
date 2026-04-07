"""
Metrics Recorder Module

Provides functionality to record, store, and export metrics from benchmark runs.
"""

import json
import csv
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable


@dataclass
class TimingMetrics:
    """Timing metrics for a single run."""
    planner_time_ms: float = 0.0
    search_time_ms: float = 0.0
    first_half_total_ms: float = 0.0

    writer_time_fast_ms: float = 0.0
    total_backend_time_fast_ms: float = 0.0

    ingest_time_ms: float = 0.0
    retrieve_time_ms: float = 0.0
    writer_time_rag_ms: float = 0.0
    total_backend_time_rag_ms: float = 0.0

    fast_web_total_ms: float = 0.0
    deep_rag_total_ms: float = 0.0


@dataclass
class CompressionMetrics:
    """Compression metrics for Deep RAG."""
    original_doc_count: int = 0
    retrieved_doc_count: int = 0
    compression_ratio: float = 0.0

    original_tokens: int = 0
    retrieved_tokens: int = 0
    compression_ratio_tokens: float = 0.0


@dataclass
class ModeComparisonResult:
    """Comparison result for a single query."""
    query_id: str
    query: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    fast_web_report: str = ""
    fast_web_metrics: Dict[str, Any] = field(default_factory=dict)

    deep_rag_report: str = ""
    deep_rag_metrics: Dict[str, Any] = field(default_factory=dict)

    timing_comparison: Dict[str, float] = field(default_factory=dict)
    winner: str = ""


class MetricsRecorder:
    """Records and exports metrics for benchmark runs."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ModeComparisonResult] = []
        self.timings: Dict[str, float] = {}

    def start_timer(self, phase_name: str, mode: str = None):
        """Start timing for a phase."""
        key = f"{mode}_{phase_name}" if mode else phase_name
        self.timings[f"{key}_start"] = time.time()

    def end_timer(self, phase_name: str, mode: str = None) -> float:
        """End timing and return duration in milliseconds."""
        key = f"{mode}_{phase_name}" if mode else phase_name
        start_key = f"{key}_start"

        if start_key not in self.timings:
            raise ValueError(f"Timer not started for {key}")

        duration_ms = (time.time() - self.timings[start_key]) * 1000
        return duration_ms

    def record_comparison_result(self, **kwargs):
        """Record a comparison result."""
        result = ModeComparisonResult(**kwargs)
        self.results.append(result)

    def export_to_csv(self, filepath: str = None) -> str:
        """Export metrics to CSV."""
        if filepath is None:
            filepath = self.output_dir / "metrics.csv"
        else:
            filepath = Path(filepath)

        if not self.results:
            print("[MetricsRecorder] No results to export")
            return ""

        fieldnames = [
            "query_id", "query", "timestamp",
            "fast_web_total_time", "fast_web_writer_time",
            "deep_rag_total_time", "deep_rag_ingest_time",
            "deep_rag_retrieve_time", "deep_rag_writer_time",
            "compression_ratio", "original_doc_count", "retrieved_doc_count"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {
                    "query_id": result.query_id,
                    "query": result.query[:50] + "..." if len(result.query) > 50 else result.query,
                    "timestamp": result.timestamp,
                    "fast_web_total_time": result.fast_web_metrics.get("total_backend_time", 0),
                    "fast_web_writer_time": result.fast_web_metrics.get("writer_time", 0),
                    "deep_rag_total_time": result.deep_rag_metrics.get("total_backend_time", 0),
                    "deep_rag_ingest_time": result.deep_rag_metrics.get("ingest_time", 0),
                    "deep_rag_retrieve_time": result.deep_rag_metrics.get("retrieve_time", 0),
                    "deep_rag_writer_time": result.deep_rag_metrics.get("writer_time", 0),
                    "compression_ratio": result.deep_rag_metrics.get("compression_ratio", 0),
                    "original_doc_count": result.deep_rag_metrics.get("original_doc_count", 0),
                    "retrieved_doc_count": result.deep_rag_metrics.get("retrieved_doc_count", 0)
                }
                writer.writerow(row)

        print(f"[MetricsRecorder] Exported CSV: {filepath}")
        return str(filepath)

    def export_to_json(self, filepath: str = None) -> str:
        """Export all results to JSON."""
        if filepath is None:
            filepath = self.output_dir / "detailed_metrics.json"
        else:
            filepath = Path(filepath)

        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_queries": len(self.results),
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "timestamp": r.timestamp,
                    "fast_web_metrics": r.fast_web_metrics,
                    "deep_rag_metrics": r.deep_rag_metrics,
                    "timing_comparison": r.timing_comparison
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[MetricsRecorder] Exported JSON: {filepath}")
        return str(filepath)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"error": "No results to summarize"}

        fast_times = [r.fast_web_metrics.get("total_backend_time", 0) for r in self.results]
        rag_times = [r.deep_rag_metrics.get("total_backend_time", 0) for r in self.results]

        fast_avg = sum(fast_times) / len(fast_times) if fast_times else 0
        rag_avg = sum(rag_times) / len(rag_times) if rag_times else 0

        compression_ratios = [
            r.deep_rag_metrics.get("compression_ratio", 0)
            for r in self.results
        ]
        avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0

        return {
            "total_queries": len(self.results),
            "fast_web": {
                "avg_total_time_ms": fast_avg,
                "min_time_ms": min(fast_times) if fast_times else 0,
                "max_time_ms": max(fast_times) if fast_times else 0
            },
            "deep_rag": {
                "avg_total_time_ms": rag_avg,
                "min_time_ms": min(rag_times) if rag_times else 0,
                "max_time_ms": max(rag_times) if rag_times else 0
            },
            "comparison": {
                "time_diff_ms": rag_avg - fast_avg,
                "time_overhead_percent": ((rag_avg - fast_avg) / fast_avg * 100) if fast_avg > 0 else 0
            },
            "compression": {
                "avg_compression_ratio": avg_compression
            }
        }
