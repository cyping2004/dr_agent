#!/usr/bin/env python3
"""Summarize LLM-as-judge results from llm_judge.json (simple metrics only)."""

import argparse
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def _avg(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _collect(results: List[Dict[str, Any]], mode: str, key: str) -> List[float]:
    return [r.get(mode, {}).get("scores", {}).get(key) for r in results]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    modes = ["fast_web", "deep_rag"]
    metrics = ["relevance", "completeness", "faithfulness"]

    summary: Dict[str, Any] = {
        "total_queries": len(results),
        "avg_scores": {},
        "wins": {m: {"fast_web": 0, "deep_rag": 0, "tie": 0} for m in metrics},
        "pairwise_wins": {"overall": {"fast_web": 0, "deep_rag": 0, "tie": 0}},
    }

    for mode in modes:
        summary["avg_scores"][mode] = {m: _avg(_collect(results, mode, m)) for m in metrics}

    for r in results:
        for m in metrics:
            fw = r.get("fast_web", {}).get("scores", {}).get(m)
            dr = r.get("deep_rag", {}).get("scores", {}).get(m)
            if fw is None or dr is None:
                continue
            if dr > fw:
                summary["wins"][m]["deep_rag"] += 1
            elif fw > dr:
                summary["wins"][m]["fast_web"] += 1
            else:
                summary["wins"][m]["tie"] += 1

        pairwise = r.get("pairwise", {})
        overall = str(pairwise.get("winner_overall", "tie")).strip().lower()
        if overall not in {"fast_web", "deep_rag", "tie"}:
            overall = "tie"
        summary["pairwise_wins"]["overall"][overall] += 1

    return summary


def write_csv(summary: Dict[str, Any], output_path: Path) -> None:
    rows = []

    avg_scores = summary.get("avg_scores", {})
    for metric in ["relevance", "completeness", "faithfulness"]:
        rows.append({
            "section": "avg_scores",
            "metric": metric,
            "fast_web": avg_scores.get("fast_web", {}).get(metric, 0),
            "deep_rag": avg_scores.get("deep_rag", {}).get(metric, 0),
            "tie": ""
        })

    wins = summary.get("wins", {})
    for metric, counts in wins.items():
        rows.append({
            "section": "wins",
            "metric": metric,
            "fast_web": counts.get("fast_web", 0),
            "deep_rag": counts.get("deep_rag", 0),
            "tie": counts.get("tie", 0)
        })

    pairwise_wins = summary.get("pairwise_wins", {})
    for metric, counts in pairwise_wins.items():
        rows.append({
            "section": "pairwise_wins",
            "metric": metric,
            "fast_web": counts.get("fast_web", 0),
            "deep_rag": counts.get("deep_rag", 0),
            "tie": counts.get("tie", 0)
        })

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "fast_web", "deep_rag", "tie"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize llm_judge.json metrics")
    parser.add_argument("--input", type=str, required=True, help="Path to llm_judge.json")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] input not found: {input_path}")
        return 1

    data = json.loads(input_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    summary = summarize(results)

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text, encoding="utf-8")
        print(f"[Done] Summary saved: {output_path}")

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        write_csv(summary, output_csv_path)
        print(f"[Done] Summary CSV saved: {output_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())