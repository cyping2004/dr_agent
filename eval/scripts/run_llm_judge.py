#!/usr/bin/env python3
"""
LLM-as-judge evaluation script.

Evaluates Fast Web and Deep RAG reports using:
- Relevance (1-5)
- Completeness (1-5)
- Faithfulness (1-5)

Also performs a pairwise winner selection between the two reports.

Usage:
    python -m eval.scripts.run_llm_judge \
        --dataset eval/datasets/basic_10.json \
        --cache-dir eval/cache/basic_10 \
        --results-dir eval/results/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from eval.utils.cache_manager import CacheManager
from dotenv import load_dotenv
load_dotenv()

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _call_llm_json(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    data = _extract_json(response.content or "")
    return data or {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: Any, low: int = 1, high: int = 5) -> int:
    score = _safe_int(value, 0)
    if low <= score <= high:
        return score
    return 0


def _score_report(
    llm: ChatOpenAI,
    query: str,
    expected_topics: List[str],
    report: str
) -> Dict[str, Any]:
    system_prompt = (
        "You are a strict evaluator. Score relevance, completeness, and faithfulness on a 1-5 scale.\n"
        "- Relevance: how well the report answers the query.\n"
        "- Completeness: how fully the report covers expected topics.\n"
        "- Faithfulness: factual accuracy and internal consistency; penalize unsupported claims.\n"
        "Return strict JSON."
    )
    topics_text = ", ".join(expected_topics) if expected_topics else "(none)"
    user_prompt = (
        f"Query: {query}\n"
        f"Expected topics: {topics_text}\n\n"
        f"Report:\n{report}\n\n"
        "Return JSON: {\"relevance\":1-5, \"completeness\":1-5, \"faithfulness\":1-5, "
        "\"rationale\":\"...\"}"
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    return {
        "scores": {
            "relevance": _clamp_score(data.get("relevance")),
            "completeness": _clamp_score(data.get("completeness")),
            "faithfulness": _clamp_score(data.get("faithfulness")),
        },
        "rationale": data.get("rationale", ""),
    }


def _compare_reports(
    llm: ChatOpenAI,
    query: str,
    expected_topics: List[str],
    fast_report: str,
    rag_report: str
) -> Dict[str, Any]:
    system_prompt = (
        "You are a strict comparative evaluator. Compare Report A (fast_web) and "
        "Report B (deep_rag). Choose a winner overall. "
        "Avoid ties unless truly equivalent. Return strict JSON."
    )
    topics_text = ", ".join(expected_topics) if expected_topics else "(none)"
    user_prompt = (
        f"Query: {query}\n"
        f"Expected topics: {topics_text}\n\n"
        "Report A (fast_web):\n"
        f"{fast_report}\n\n"
        "Report B (deep_rag):\n"
        f"{rag_report}\n\n"
        "Return JSON: {\"winner_overall\":\"fast_web|deep_rag|tie\", "
        "\"rationale\":\"...\", \"confidence\":0-1}"
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    winner = str(data.get("winner_overall", "tie")).strip().lower()
    if winner not in {"fast_web", "deep_rag", "tie"}:
        winner = "tie"
    confidence = data.get("confidence", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "winner_overall": winner,
        "rationale": data.get("rationale", ""),
        "confidence": confidence,
    }


def _load_dataset(dataset_path: str) -> Dict[str, Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[str, Dict[str, Any]] = {}
    for q in data.get("queries", []):
        qid = q.get("id")
        if qid:
            mapping[qid] = q
    return mapping


def _read_report(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation")
    parser.add_argument("--dataset", type=str, default="eval/datasets/basic_10.json")
    parser.add_argument("--cache-dir", type=str, default="eval/cache/basic_10")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[Error] OPENAI_API_KEY not set")
        return 1

    dataset = _load_dataset(args.dataset)
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    cached_outputs = cache_manager.load_all_cached_queries_with_ids()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[Error] results dir not found: {results_dir}")
        return 1

    output_path = Path(args.output) if args.output else results_dir / "llm_judge.json"

    judge_model = (
        args.judge_model
        or os.getenv("JUDGE_MODEL")
    )

    llm = ChatOpenAI(
        model=judge_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0
    )

    all_results = []

    for query_id, output in cached_outputs:
        query = output.query
        expected_topics = dataset.get(query_id, {}).get("expected_topics", [])

        fast_report_path = results_dir / "reports" / f"{query_id}_fast_web.md"
        rag_report_path = results_dir / "reports" / f"{query_id}_deep_rag.md"

        fast_report = _read_report(fast_report_path)
        rag_report = _read_report(rag_report_path)

        if not fast_report or not rag_report:
            print(f"[Warn] Missing report for {query_id}, skipping")
            continue

        fast_eval = _score_report(
            llm=llm,
            query=query,
            expected_topics=expected_topics,
            report=fast_report
        )

        rag_eval = _score_report(
            llm=llm,
            query=query,
            expected_topics=expected_topics,
            report=rag_report
        )

        pairwise = _compare_reports(
            llm=llm,
            query=query,
            expected_topics=expected_topics,
            fast_report=fast_report,
            rag_report=rag_report
        )

        all_results.append({
            "query_id": query_id,
            "query": query,
            "expected_topics": expected_topics,
            "fast_web": fast_eval,
            "deep_rag": rag_eval,
            "pairwise": pairwise,
        })

        print(f"[Evaluated] {query_id}")

    payload = {
        "run_metadata": {
            "dataset": args.dataset,
            "results_dir": str(results_dir),
            "judge_model": judge_model,
            "metrics": ["relevance", "completeness", "faithfulness"],
        },
        "results": all_results,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] LLM judge results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())