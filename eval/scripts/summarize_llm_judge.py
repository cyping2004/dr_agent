#!/usr/bin/env python3
"""Summarize LLM-as-judge results from llm_judge.json."""

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


def _collect_ratio(results: List[Dict[str, Any]], mode: str) -> List[float]:
    return [r.get(mode, {}).get("faithfulness_ratio") for r in results]


def _collect_claims(results: List[Dict[str, Any]], mode: str, key: str) -> List[int]:
    return [r.get(mode, {}).get("claim_stats", {}).get(key, 0) for r in results]


def _collect_coverage(results: List[Dict[str, Any]], mode: str) -> List[str]:
    statuses: List[str] = []
    for r in results:
        coverage = r.get(mode, {}).get("coverage", [])
        if not isinstance(coverage, list):
            continue
        for item in coverage:
            status = str(item.get("status", "")).strip().lower()
            if status:
                statuses.append(status)
    return statuses


def _collect_race_scores(results: List[Dict[str, Any]], mode: str, key: str) -> List[float]:
    values: List[float] = []
    for r in results:
        race = r.get("race", {}) or {}
        scores = race.get(mode, {}) or {}
        values.append(scores.get(key))
    return values


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    modes = ["fast_web", "deep_rag"]
    metrics = ["relevance", "completeness", "coherence", "faithfulness"]

    summary: Dict[str, Any] = {
        "total_queries": len(results),
        "avg_scores": {},
        "avg_faithfulness_ratio": {},
        "claim_totals": {},
        "coverage_totals": {},
        "wins": {m: {"fast_web": 0, "deep_rag": 0, "tie": 0} for m in metrics},
        "pairwise_wins": {
            "overall": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "relevance": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "completeness": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "coherence": {"fast_web": 0, "deep_rag": 0, "tie": 0},
        },
        "race_avg_scores": {},
        "race_pairwise_wins": {
            "overall": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "comprehensiveness": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "insight_depth": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "instruction_following": {"fast_web": 0, "deep_rag": 0, "tie": 0},
            "readability": {"fast_web": 0, "deep_rag": 0, "tie": 0},
        },
    }

    for mode in modes:
        summary["avg_scores"][mode] = {m: _avg(_collect(results, mode, m)) for m in metrics}
        summary["avg_faithfulness_ratio"][mode] = _avg(_collect_ratio(results, mode))
        summary["claim_totals"][mode] = {
            "supported": int(sum(_collect_claims(results, mode, "supported"))),
            "contradicted": int(sum(_collect_claims(results, mode, "contradicted"))),
            "insufficient": int(sum(_collect_claims(results, mode, "insufficient"))),
            "total": int(sum(_collect_claims(results, mode, "total"))),
        }

        coverage = _collect_coverage(results, mode)
        covered = sum(1 for s in coverage if s == "covered")
        partial = sum(1 for s in coverage if s == "partial")
        missing = sum(1 for s in coverage if s == "missing")
        total = covered + partial + missing
        ratio = ((covered + 0.5 * partial) / total) if total else 0.0
        summary["coverage_totals"][mode] = {
            "covered": covered,
            "partial": partial,
            "missing": missing,
            "total": total,
            "coverage_ratio": round(ratio, 4),
        }

        summary["race_avg_scores"][mode] = {
            "comprehensiveness": _avg(_collect_race_scores(results, mode, "comprehensiveness")),
            "insight_depth": _avg(_collect_race_scores(results, mode, "insight_depth")),
            "instruction_following": _avg(_collect_race_scores(results, mode, "instruction_following")),
            "readability": _avg(_collect_race_scores(results, mode, "readability")),
        }

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

        dim = pairwise.get("dimension_winners", {}) or {}
        for key in ["relevance", "completeness", "coherence"]:
            winner = str(dim.get(key, "tie")).strip().lower()
            if winner not in {"fast_web", "deep_rag", "tie"}:
                winner = "tie"
            summary["pairwise_wins"][key][winner] += 1

        race = r.get("race", {}) or {}
        race_pairwise = race.get("pairwise", {}) or {}
        overall = str(race_pairwise.get("winner_overall", "tie")).strip().lower()
        if overall not in {"fast_web", "deep_rag", "tie"}:
            overall = "tie"
        summary["race_pairwise_wins"]["overall"][overall] += 1

        dim = race_pairwise.get("dimension_winners", {}) or {}
        for key in ["comprehensiveness", "insight_depth", "instruction_following", "readability"]:
            winner = str(dim.get(key, "tie")).strip().lower()
            if winner not in {"fast_web", "deep_rag", "tie"}:
                winner = "tie"
            summary["race_pairwise_wins"][key][winner] += 1

    return summary


def write_csv(summary: Dict[str, Any], output_path: Path) -> None:
    rows = []

    avg_scores = summary.get("avg_scores", {})
    for metric in ["relevance", "completeness", "coherence", "faithfulness"]:
        rows.append({
            "section": "avg_scores",
            "metric": metric,
            "fast_web": avg_scores.get("fast_web", {}).get(metric, 0),
            "deep_rag": avg_scores.get("deep_rag", {}).get(metric, 0),
            "tie": ""
        })

    rows.append({
        "section": "avg_ratio",
        "metric": "faithfulness_ratio",
        "fast_web": summary.get("avg_faithfulness_ratio", {}).get("fast_web", 0),
        "deep_rag": summary.get("avg_faithfulness_ratio", {}).get("deep_rag", 0),
        "tie": ""
    })

    claim_totals = summary.get("claim_totals", {})
    for metric in ["supported", "contradicted", "insufficient", "total"]:
        rows.append({
            "section": "claim_totals",
            "metric": metric,
            "fast_web": claim_totals.get("fast_web", {}).get(metric, 0),
            "deep_rag": claim_totals.get("deep_rag", {}).get(metric, 0),
            "tie": ""
        })

    coverage_totals = summary.get("coverage_totals", {})
    for metric in ["covered", "partial", "missing", "total", "coverage_ratio"]:
        rows.append({
            "section": "coverage_totals",
            "metric": metric,
            "fast_web": coverage_totals.get("fast_web", {}).get(metric, 0),
            "deep_rag": coverage_totals.get("deep_rag", {}).get(metric, 0),
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

    race_avg_scores = summary.get("race_avg_scores", {})
    for metric in ["comprehensiveness", "insight_depth", "instruction_following", "readability"]:
        rows.append({
            "section": "race_avg_scores",
            "metric": metric,
            "fast_web": race_avg_scores.get("fast_web", {}).get(metric, 0),
            "deep_rag": race_avg_scores.get("deep_rag", {}).get(metric, 0),
            "tie": ""
        })

    race_pairwise = summary.get("race_pairwise_wins", {})
    for metric, counts in race_pairwise.items():
        rows.append({
            "section": "race_pairwise_wins",
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
