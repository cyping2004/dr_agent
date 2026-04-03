"""BM25 sparse retrieval utilities.

This module implements a lightweight BM25 (Okapi) index without external
dependencies, designed for small-to-medium local corpora.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Dict, Iterable, List, Tuple

from langchain_core.documents import Document


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def default_tokenize(text: str) -> List[str]:
    """Tokenize for mixed Chinese/English text.

    - English/numbers: lowercased word tokens
    - CJK: individual characters (simple but dependency-free)

    This is intentionally simple; if you need better Chinese segmentation,
    consider plugging in a jieba-based tokenizer.
    """

    if not text:
        return []

    tokens: List[str] = []

    # Grab alnum word tokens
    for m in _ALNUM_RE.finditer(text):
        w = m.group(0).lower()
        if w:
            tokens.append(w)

    # Add CJK characters as tokens
    for ch in text:
        if _CJK_RE.match(ch):
            tokens.append(ch)

    return tokens


def doc_key(doc: Document) -> str:
    """Generate a stable key for a document for dedup & fusion."""

    source = ""
    try:
        source = str((doc.metadata or {}).get("source", ""))
    except Exception:
        source = ""

    payload = (source + "\n" + (doc.page_content or "")).encode("utf-8", errors="ignore")
    return sha1(payload).hexdigest()


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    """A simple BM25 Okapi index."""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = default_tokenize,
        config: BM25Config | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or BM25Config()

        self._docs: List[Document] = []
        self._doc_keys: List[str] = []
        self._tf: List[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._idf: Dict[str, float] = {}
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._docs)

    def build(self, docs: Iterable[Document]) -> None:
        self._docs = list(docs)
        self._doc_keys = [doc_key(d) for d in self._docs]

        self._tf = []
        self._df = Counter()
        self._doc_len = []

        for doc in self._docs:
            tokens = self.tokenizer(doc.page_content or "")
            tf = Counter(tokens)
            self._tf.append(tf)
            dl = sum(tf.values())
            self._doc_len.append(dl)

            for term in tf.keys():
                self._df[term] += 1

        n_docs = len(self._docs)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0

        # BM25 idf with smoothing; common variant
        self._idf = {}
        for term, df in self._df.items():
            # idf = log((N - df + 0.5)/(df + 0.5) + 1)
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if not self._docs:
            return []

        q_terms = self.tokenizer(query)
        if not q_terms:
            return []

        q_tf = Counter(q_terms)
        k1 = self.config.k1
        b = self.config.b

        scores: List[Tuple[int, float]] = []
        for idx, tf in enumerate(self._tf):
            dl = self._doc_len[idx] or 0
            denom_norm = 1.0
            if self._avgdl > 0:
                denom_norm = 1.0 - b + b * (dl / self._avgdl)

            score = 0.0
            for term, qcnt in q_tf.items():
                if term not in tf:
                    continue
                f = tf[term]
                idf = self._idf.get(term, 0.0)
                # BM25 term score
                term_score = idf * (f * (k1 + 1.0)) / (f + k1 * denom_norm)
                # query term frequency multiplier (optional; keep linear)
                score += term_score * float(qcnt)

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[Tuple[Document, float]] = []
        for idx, score in scores[: max(top_k, 0)]:
            results.append((self._docs[idx], float(score)))
        return results

    def get_doc_by_key(self) -> Dict[str, Document]:
        return {k: d for k, d in zip(self._doc_keys, self._docs)}
