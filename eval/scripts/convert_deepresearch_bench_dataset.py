#!/usr/bin/env python3
"""Convert DeepResearch Bench query.jsonl to local dataset JSON format."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_query(row: Dict[str, Any]) -> str:
    return (
        row.get("prompt")
        or row.get("query")
        or row.get("question")
        or row.get("instruction")
        or ""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert DeepResearch Bench dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to query.jsonl")
    parser.add_argument("--output", type=str, default="eval/datasets/deepresearch_bench.json")
    parser.add_argument("--dataset-name", type=str, default="deepresearch_bench")
    parser.add_argument("--version", type=str, default="1.0")
    parser.add_argument("--description", type=str, default="DeepResearch Bench query set")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] input not found: {input_path}")
        return 1

    rows = _read_jsonl(input_path)
    if args.limit:
        rows = rows[: args.limit]

    queries = []
    for row in rows:
        qid = row.get("id") or row.get("query_id") or row.get("qid")
        query = _get_query(row)
        if not qid or not query:
            continue
        queries.append({
            "id": str(qid),
            "query": str(query),
            "expected_topics": [],
            "difficulty": row.get("difficulty", ""),
            "tags": row.get("tags", []),
        })

    payload = {
        "dataset_name": args.dataset_name,
        "version": args.version,
        "description": args.description,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "queries": queries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] Wrote {len(queries)} queries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
