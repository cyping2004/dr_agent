#!/usr/bin/env python3
"""
LLM-as-judge evaluation script.

Evaluates Fast Web and Deep RAG reports using:
- Relevance, Completeness, Coherence (1-5)
- Faithfulness (claim-level evidence support)

Usage:
    python -m eval.scripts.run_llm_judge \
        --dataset eval/datasets/basic_10.json \
        --cache-dir eval/cache/basic_10 \
        --results-dir eval/results/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from eval.utils.cache_manager import CacheManager
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents, embed_query
from ingestion.vector_store import VectorStore

RACE_STATIC_CRITERIA = {
    "comprehensiveness": [
        {"criterion": "关键子主题覆盖", "explanation": "是否覆盖任务中所有关键子主题与核心方面", "weight": 0.3},
        {"criterion": "信息深度与细节", "explanation": "是否提供必要的细节、背景与解释，而非浅层罗列", "weight": 0.3},
        {"criterion": "证据与案例支撑", "explanation": "是否有数据、事实、案例或引用支撑核心结论", "weight": 0.2},
        {"criterion": "多角度与平衡性", "explanation": "是否考虑多个视角与潜在反例/限制", "weight": 0.2},
    ],
    "insight_depth": [
        {"criterion": "因果与机制分析", "explanation": "是否解释原因、机制或逻辑链条", "weight": 0.3},
        {"criterion": "综合与抽象能力", "explanation": "是否将分散信息整合成更高层次结论", "weight": 0.25},
        {"criterion": "权衡与影响分析", "explanation": "是否分析利弊、风险与影响", "weight": 0.25},
        {"criterion": "可行动结论", "explanation": "是否给出可执行或可参考的结论/建议", "weight": 0.2},
    ],
    "instruction_following": [
        {"criterion": "问题回应完整性", "explanation": "是否完整回应任务所有明确问题", "weight": 0.4},
        {"criterion": "范围与约束遵守", "explanation": "是否遵守时间/地域/对象等限定", "weight": 0.35},
        {"criterion": "聚焦与相关性", "explanation": "是否紧扣任务，避免无关扩写", "weight": 0.25},
    ],
    "readability": [
        {"criterion": "结构与层次", "explanation": "是否结构清晰、层次分明", "weight": 0.3},
        {"criterion": "语言清晰度", "explanation": "语言是否准确、流畅、易理解", "weight": 0.3},
        {"criterion": "信息组织与重点突出", "explanation": "是否突出要点、组织有序", "weight": 0.25},
        {"criterion": "格式与可视化", "explanation": "格式是否规范，图表/列表使用得当", "weight": 0.15},
    ],
}


def _sanitize_collection_suffix(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    safe = safe.strip("._-")
    return safe or "query"


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


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


def _extract_claims(llm: ChatOpenAI, report: str, max_claims: int) -> List[str]:
    system_prompt = (
        "You extract atomic factual claims from a report. "
        "Return strict JSON: {\"claims\": [..]}"
    )
    user_prompt = (
        f"Report:\n{report}\n\n"
        f"Extract up to {max_claims} atomic, checkable claims. "
        "Return JSON with a 'claims' list of strings."
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        return []
    return [str(c).strip() for c in claims if str(c).strip()]


def _score_quality(
    llm: ChatOpenAI,
    query: str,
    expected_topics: List[str],
    report: str
) -> Dict[str, Any]:
    system_prompt = (
        "You are a strict evaluator. Use this rubric:\n"
        "- Relevance: 1=off-topic, 3=partially answers, 5=fully answers the query.\n"
        "- Completeness: 1=misses most expected topics, 3=some covered, 5=all covered with detail.\n"
        "- Coherence: 1=disorganized, 3=mostly coherent with minor issues, 5=highly coherent.\n"
        "Do not give 5 unless clearly satisfied. Return strict JSON."
    )
    topics_text = ", ".join(expected_topics) if expected_topics else "(none)"
    user_prompt = (
        f"Query: {query}\n"
        f"Expected topics: {topics_text}\n\n"
        f"Report:\n{report}\n\n"
        "Return JSON: {\"relevance\":1-5, \"completeness\":1-5, "
        "\"coherence\":1-5, \"rationale\":\"...\", "
        "\"coverage\":[{\"topic\":\"...\", \"status\":\"covered|partial|missing\"}]}"
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    return {
        "relevance": _safe_int(data.get("relevance", 0) or 0),
        "completeness": _safe_int(data.get("completeness", 0) or 0),
        "coherence": _safe_int(data.get("coherence", 0) or 0),
        "rationale": data.get("rationale", ""),
        "coverage": data.get("coverage", []),
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
        "Report B (deep_rag). Choose a winner per metric and overall. "
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
        "Return JSON: {"
        "\"winner_overall\":\"fast_web|deep_rag|tie\", "
        "\"dimension_winners\":{\"relevance\":\"fast_web|deep_rag|tie\", "
        "\"completeness\":\"fast_web|deep_rag|tie\", "
        "\"coherence\":\"fast_web|deep_rag|tie\"}, "
        "\"rationale\":\"...\", \"confidence\":0-1}"
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    winners = data.get("dimension_winners", {}) or {}
    return {
        "winner_overall": data.get("winner_overall", "tie"),
        "dimension_winners": {
            "relevance": winners.get("relevance", "tie"),
            "completeness": winners.get("completeness", "tie"),
            "coherence": winners.get("coherence", "tie"),
        },
        "rationale": data.get("rationale", ""),
        "confidence": data.get("confidence", 0),
    }


def _score_race(
    llm: ChatOpenAI,
    query: str,
    report: str
) -> Dict[str, Any]:
    criteria_list = json.dumps(RACE_STATIC_CRITERIA, ensure_ascii=False, indent=2)
    system_prompt = (
        "<system_role>你是一名严格、细致、客观的调研文章评估专家。"
        "你擅长根据给定评判标准，对文章进行逐条评估并给出分数。</system_role>"
    )
    user_prompt = (
        "<user_prompt>\n"
        "**任务背景**\n"
        "有一个深度调研任务，需要评估一篇文章的质量。\n"
        f"<task>\n{query}\n</task>\n\n"
        "**待评估文章**\n"
        f"<article>\n{report}\n</article>\n\n"
        "**评估标准**\n"
        "请根据以下四个维度的评判标准逐条评估并打分。\n"
        f"<criteria_list>\n{criteria_list}\n</criteria_list>\n\n"
        "<Instruction>\n"
        "1. 对每个维度的每条标准逐条分析，并给出0-10连续分数。\n"
        "2. 输出四个维度的总分（0-10），并给出overall_score。\n"
        "3. 分数规则：0-2很差，2-4较差，4-6中等，6-8较好，8-10优秀。\n"
        "4. 仅输出JSON，不要额外文本。\n"
        "</Instruction>\n\n"
        "<output_format>\n"
        "{\n"
        "  \"comprehensiveness\": 0-10,\n"
        "  \"insight_depth\": 0-10,\n"
        "  \"instruction_following\": 0-10,\n"
        "  \"readability\": 0-10,\n"
        "  \"overall_score\": 0-10,\n"
        "  \"details\": {\n"
        "    \"comprehensiveness\": [{\"criterion\": \"...\", \"score\": 0-10, \"analysis\": \"...\"}],\n"
        "    \"insight_depth\": [{\"criterion\": \"...\", \"score\": 0-10, \"analysis\": \"...\"}],\n"
        "    \"instruction_following\": [{\"criterion\": \"...\", \"score\": 0-10, \"analysis\": \"...\"}],\n"
        "    \"readability\": [{\"criterion\": \"...\", \"score\": 0-10, \"analysis\": \"...\"}]\n"
        "  }\n"
        "}\n"
        "</output_format>\n"
        "</user_prompt>"
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    return {
        "comprehensiveness": _safe_int(data.get("comprehensiveness", 0) or 0),
        "insight_depth": _safe_int(data.get("insight_depth", 0) or 0),
        "instruction_following": _safe_int(data.get("instruction_following", 0) or 0),
        "readability": _safe_int(data.get("readability", 0) or 0),
        "overall_score": _safe_int(data.get("overall_score", 0) or 0),
        "details": data.get("details", {}),
    }


def _compare_reports_race(
    llm: ChatOpenAI,
    query: str,
    fast_report: str,
    rag_report: str
) -> Dict[str, Any]:
    criteria_list = json.dumps(RACE_STATIC_CRITERIA, ensure_ascii=False, indent=2)
    system_prompt = (
        "<system_role>你是一名严格、细致、客观的调研文章评估专家。"
        "你擅长根据评判标准对两篇文章进行对比评估并给出分数。</system_role>"
    )
    user_prompt = (
        "<user_prompt>\n"
        "**任务背景**\n"
        f"<task>\n{query}\n</task>\n\n"
        "**待评估文章**\n"
        f"<article_1>\n{fast_report}\n</article_1>\n\n"
        f"<article_2>\n{rag_report}\n</article_2>\n\n"
        "**评估标准**\n"
        f"<criteria_list>\n{criteria_list}\n</criteria_list>\n\n"
        "<Instruction>\n"
        "1. 逐条标准对比分析，并给出两篇文章各自的0-10分。\n"
        "2. 给出每个维度的胜者与overall胜者。\n"
        "3. 尽量避免平局，除非确实等同。\n"
        "4. 仅输出JSON，不要额外文本。\n"
        "</Instruction>\n\n"
        "<output_format>\n"
        "{\n"
        "  \"winner_overall\": \"fast_web|deep_rag|tie\",\n"
        "  \"dimension_winners\": {\n"
        "    \"comprehensiveness\": \"fast_web|deep_rag|tie\",\n"
        "    \"insight_depth\": \"fast_web|deep_rag|tie\",\n"
        "    \"instruction_following\": \"fast_web|deep_rag|tie\",\n"
        "    \"readability\": \"fast_web|deep_rag|tie\"\n"
        "  },\n"
        "  \"article_scores\": {\n"
        "    \"fast_web\": 0-10,\n"
        "    \"deep_rag\": 0-10\n"
        "  },\n"
        "  \"details\": {\n"
        "    \"comprehensiveness\": [{\"criterion\": \"...\", \"analysis\": \"...\", \"article_1_score\": 0-10, \"article_2_score\": 0-10}],\n"
        "    \"insight_depth\": [{\"criterion\": \"...\", \"analysis\": \"...\", \"article_1_score\": 0-10, \"article_2_score\": 0-10}],\n"
        "    \"instruction_following\": [{\"criterion\": \"...\", \"analysis\": \"...\", \"article_1_score\": 0-10, \"article_2_score\": 0-10}],\n"
        "    \"readability\": [{\"criterion\": \"...\", \"analysis\": \"...\", \"article_1_score\": 0-10, \"article_2_score\": 0-10}]\n"
        "  }\n"
        "}\n"
        "</output_format>\n"
        "</user_prompt>"
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    winners = data.get("dimension_winners", {}) or {}
    return {
        "winner_overall": data.get("winner_overall", "tie"),
        "dimension_winners": {
            "comprehensiveness": winners.get("comprehensiveness", "tie"),
            "insight_depth": winners.get("insight_depth", "tie"),
            "instruction_following": winners.get("instruction_following", "tie"),
            "readability": winners.get("readability", "tie"),
        },
        "article_scores": data.get("article_scores", {}),
        "details": data.get("details", {}),
        "rationale": data.get("rationale", ""),
        "confidence": data.get("confidence", 0),
    }


def _format_evidence(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {}) or {}
        url = meta.get("url", "")
        title = meta.get("title", "")
        header_parts = []
        if title:
            header_parts.append(f"title={title}")
        if url:
            header_parts.append(f"url={url}")
        header = " | ".join(header_parts)
        body = chunk.get("text", "")[:max_chars]
        if header:
            lines.append(f"[{idx}] {header}\n{body}")
        else:
            lines.append(f"[{idx}] {body}")
    return "\n\n".join(lines)


def _judge_claim(
    llm: ChatOpenAI,
    claim: str,
    evidence_text: str
) -> Dict[str, Any]:
    system_prompt = (
        "You judge whether a claim is supported by the evidence. "
        "Return strict JSON with verdict supported|contradicted|insufficient."
    )
    user_prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Return JSON: {\"verdict\":\"supported|contradicted|insufficient\", "
        "\"confidence\":0-1, \"rationale\":\"...\"}."
    )
    data = _call_llm_json(llm, system_prompt, user_prompt)
    verdict = str(data.get("verdict", "insufficient")).strip().lower()
    if verdict not in {"supported", "contradicted", "insufficient"}:
        verdict = "insufficient"
    confidence = data.get("confidence", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": data.get("rationale", ""),
    }


def _retrieve_evidence(
    vector_store: VectorStore,
    claim: str,
    top_k: int
) -> List[Dict[str, Any]]:
    embedding = embed_query(claim)
    pairs = vector_store.similarity_search_with_score(embedding, top_k=top_k)
    evidence = []
    for doc, score in pairs:
        evidence.append({
            "text": doc.page_content,
            "metadata": doc.metadata or {},
            "score": score,
        })
    return evidence


def _evaluate_report(
    llm: ChatOpenAI,
    report: str,
    query: str,
    expected_topics: List[str],
    vector_store: VectorStore,
    max_claims: int,
    evidence_top_k: int,
    max_evidence_chars: int
) -> Dict[str, Any]:
    claims = _extract_claims(llm, report, max_claims=max_claims)
    claim_results = []
    supported = contradicted = insufficient = 0

    for claim in claims:
        evidence_chunks = _retrieve_evidence(vector_store, claim, top_k=evidence_top_k)
        evidence_text = _format_evidence(evidence_chunks, max_chars=max_evidence_chars)
        judge = _judge_claim(llm, claim, evidence_text)
        verdict = judge.get("verdict")
        if verdict == "supported":
            supported += 1
        elif verdict == "contradicted":
            contradicted += 1
        else:
            insufficient += 1
        claim_results.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": judge.get("confidence", 0),
            "rationale": judge.get("rationale", ""),
            "evidence_count": len(evidence_chunks),
        })

    total = len(claims) or 0
    faithfulness = (supported / total) if total else 0.0

    quality = _score_quality(llm, query=query, expected_topics=expected_topics, report=report)
    scores = {
        "relevance": quality.get("relevance", 0),
        "completeness": quality.get("completeness", 0),
        "coherence": quality.get("coherence", 0),
        "faithfulness": round(faithfulness * 5, 2),
    }

    return {
        "scores": scores,
        "faithfulness_ratio": faithfulness,
        "claim_stats": {
            "total": total,
            "supported": supported,
            "contradicted": contradicted,
            "insufficient": insufficient,
        },
        "coverage": quality.get("coverage", []),
        "claims": claim_results,
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
    parser.add_argument("--max-claims", type=int, default=8)
    parser.add_argument("--evidence-top-k", type=int, default=5)
    parser.add_argument("--max-evidence-chars", type=int, default=800)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--disable-race", action="store_true", help="Disable RACE-style scoring")
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
    enable_race = not args.disable_race

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

        safe_query = _sanitize_collection_suffix(query_id)
        collection_name = f"judge_{safe_query}_{int(time.time())}"
        vector_store = VectorStore(collection_name=collection_name)

        try:
            chunked = chunk_documents(
                output.documents,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap
            )
            embedded_pairs = embed_documents(chunked)
            texts = [doc.page_content for doc, _ in embedded_pairs]
            metadatas = [doc.metadata for doc, _ in embedded_pairs]
            embeddings = [embedding for _, embedding in embedded_pairs]
            vector_store.upsert(texts, embeddings, metadatas=metadatas)

            fast_eval = _evaluate_report(
                llm=llm,
                report=fast_report,
                query=query,
                expected_topics=expected_topics,
                vector_store=vector_store,
                max_claims=args.max_claims,
                evidence_top_k=args.evidence_top_k,
                max_evidence_chars=args.max_evidence_chars
            )

            rag_eval = _evaluate_report(
                llm=llm,
                report=rag_report,
                query=query,
                expected_topics=expected_topics,
                vector_store=vector_store,
                max_claims=args.max_claims,
                evidence_top_k=args.evidence_top_k,
                max_evidence_chars=args.max_evidence_chars
            )

            race_fast = _score_race(llm, query=query, report=fast_report) if enable_race else {}
            race_rag = _score_race(llm, query=query, report=rag_report) if enable_race else {}
            race_pairwise = (
                _compare_reports_race(llm, query=query, fast_report=fast_report, rag_report=rag_report)
                if enable_race else {}
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
                "race": {
                    "fast_web": race_fast,
                    "deep_rag": race_rag,
                    "pairwise": race_pairwise,
                } if enable_race else {},
            })

        finally:
            try:
                vector_store.client.delete_collection(name=collection_name)
            except Exception:
                pass

    payload = {
        "run_metadata": {
            "dataset": args.dataset,
            "results_dir": str(results_dir),
            "judge_model": judge_model,
            "max_claims": args.max_claims,
            "evidence_top_k": args.evidence_top_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "race_enabled": enable_race,
        },
        "results": all_results
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] LLM judge results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
