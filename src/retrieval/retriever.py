# src/retrieval/retriever.py
"""
Retriever — finds the most relevant chunks for a given query.

Retrieval modes:
  dense   — pure vector similarity (ChromaDB cosine)
  hybrid  — dense + BM25, blended by `alpha` with min-max normalization
  rerank  — hybrid then cross-encoder reranker (most accurate, slowest)

Key features:
  • Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
  • Min-max score normalization before hybrid blending
  • Financial tokenizer (preserves 6.25%, GNPA, NIM, etc.)
  • Query understanding (detects RBI, company, fiscal year)
  • Where-filter and scope matching on BM25 candidates
  • Near-duplicate suppression (MMR-style text deduplication)
  • LRU embedding cache for fast repeated queries
  • Alpha sweep helper: retrieve_sweep(query, alphas=[0.5, 0.7, 0.9])
"""

import logging
import os
import re
import time
from typing import Optional

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.security import scope_allows

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize_financial_text(text: str) -> list[str]:
    """
    Tokenize query and passage text while preserving financial figures,
    percentages (6.25%), ratios, and acronyms (GNPA, CASA, FY24).
    """
    pattern = r"\b\d+(?:\.\d+)?%?|\b[a-zA-Z0-9&/_-]+\b"
    return [t.lower() for t in re.findall(pattern, text)]


def understand_query(query: str) -> dict[str, Optional[str]]:
    """
    Query understanding: extract entity, doc_type, and fiscal_year from the question.

    Helps avoid cross-company confusion (e.g. repo rate question citing L&T).
    """
    q_lower = query.lower()
    inferred: dict[str, Optional[str]] = {
        "company": None,
        "doc_type": None,
        "fiscal_year": None,
    }

    # Macroeconomic / RBI indicators
    rbi_triggers = [
        "repo rate", "reverse repo", "cash reserve ratio", "crr", "statutory liquidity",
        "slr", "monetary policy", "rbi", "governor", "mpc", "inflation target",
        "headline inflation", "cpi inflation",
    ]
    if any(trigger in q_lower for trigger in rbi_triggers):
        inferred["doc_type"] = "rbi"
        inferred["company"] = "RBI"

    # Company entities
    company_map = {
        "hdfc": "HDFC Bank",
        "tcs": "TCS",
        "infosys": "Infosys",
        "infy": "Infosys",
        "wipro": "Wipro",
        "hcl": "HCL Technologies",
        "icici": "ICICI Bank",
        "axis": "Axis Bank",
        "sbi": "SBI",
        "state bank": "SBI",
        "reliance": "Reliance",
        "bajaj": "Bajaj Finance",
        "asian paints": "Asian Paints",
        "l&t": "L&T",
        "larsen": "L&T",
        "maruti": "Maruti Suzuki",
        "sun pharma": "Sun Pharma",
        "itc": "ITC",
    }
    for kw, comp in company_map.items():
        if re.search(rf"\b{re.escape(kw)}\b", q_lower):
            inferred["company"] = comp
            inferred["doc_type"] = "company"
            break

    # Year detection
    yr_match = re.search(r"\b(20\d{2}|fy\d{2,4})\b", q_lower)
    if yr_match:
        inferred["fiscal_year"] = yr_match.group(0).upper()

    return inferred


def matches_where_filter(chunk: dict, where_filter: Optional[dict]) -> bool:
    """Check if chunk metadata satisfies a ChromaDB-style where_filter."""
    if not where_filter:
        return True

    if "$and" in where_filter:
        return all(matches_where_filter(chunk, sub) for sub in where_filter["$and"])

    for key, expected in where_filter.items():
        if key == "$and":
            continue
        actual = chunk.get(key)
        if isinstance(expected, dict):
            # Handle {"$eq": val} or {"$in": [...]}
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False

    return True


def deduplicate_chunks(chunks: list[dict], overlap_threshold: float = 0.85) -> list[dict]:
    """
    Drop near-duplicate chunks so the top-K isn't multiple copies of one passage.
    Compares 3-gram sets of candidate texts.
    """
    unique: list[dict] = []
    seen_ngrams: list[set[str]] = []

    for chunk in chunks:
        words = chunk["text"].lower().split()
        if len(words) < 5:
            unique.append(chunk)
            continue

        ngrams = {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}
        is_dup = False
        for seen in seen_ngrams:
            intersection = len(ngrams & seen)
            union = len(ngrams | seen)
            if union > 0 and (intersection / union) > overlap_threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(chunk)
            seen_ngrams.append(ngrams)

    return unique


class FinanceRetriever:
    """
    Retrieves relevant chunks from ChromaDB for a given query.

    Args:
        collection:     ChromaDB collection.
        model:          Bi-encoder (SentenceTransformer) for dense search.
        top_k:          Final number of chunks returned.
        reranker:       Optional CrossEncoder.
        rerank_factor:  How many candidates to feed the reranker (default 4x).
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        model: SentenceTransformer,
        top_k: int = 5,
        reranker: Optional[CrossEncoder] = None,
        rerank_factor: int = 4,
    ):
        self.collection     = collection
        self.model          = model
        self.top_k          = top_k
        self.reranker       = reranker
        self.rerank_factor  = rerank_factor
        self._embed_cache: dict[str, list[float]] = {}
        self.min_score_threshold: float = 0.20

    # ---------------------------------------------------------------------- #
    # Internals                                                                #
    # ---------------------------------------------------------------------- #

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed a query with the BGE query prefix and in-memory LRU caching.
        """
        if query in self._embed_cache:
            return self._embed_cache[query]

        prefixed = f"Represent this query for searching relevant passages: {query}"
        emb = self.model.encode(
            [prefixed], normalize_embeddings=True
        ).tolist()[0]

        # In-memory LRU cache cap
        if len(self._embed_cache) >= 1024:
            self._embed_cache.pop(next(iter(self._embed_cache)))
        self._embed_cache[query] = emb
        return emb

    def _fetch_dense(
        self,
        query: str,
        n: int,
        where_filter: Optional[dict],
    ) -> list[dict]:
        """Raw ChromaDB vector search; returns up to `n` chunks."""
        embedding = self._embed_query(query)
        n_safe = min(n, self.collection.count())
        if n_safe == 0:
            return []

        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results":        n_safe,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)
        if not results["documents"] or not results["documents"][0]:
            return []

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            md = meta if isinstance(meta, dict) else {}
            chunks.append({
                "text":        doc,
                "score":       round(1.0 - dist, 4),
                "source_file": md.get("source_file", ""),
                "page_number": md.get("page_number", 0),
                "company":     md.get("company", ""),
                "doc_type":    md.get("doc_type", ""),
                "fiscal_year": md.get("fiscal_year", ""),
                "report_type": md.get("report_type", ""),
            })
        return chunks

    def _apply_bm25(
        self,
        query: str,
        candidates: list[dict],
        alpha: float,
        scope: Optional[dict],
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Score `candidates` with BM25, apply min-max normalization, blend with
        dense scores at `alpha`, apply scope and metadata filters, and sort by hybrid_score.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — skipping BM25 blend")
            out = [
                {**c} for c in candidates
                if (scope is None or scope_allows(scope, c))
                and matches_where_filter(c, where_filter)
            ]
            return out

        # Copy candidates so callers can safely reuse candidate lists
        candidates = [{**c} for c in candidates]
        tokenized_corpus = [tokenize_financial_text(c["text"]) for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize_financial_text(query)
        raw_scores = bm25.get_scores(query_tokens)

        # Normalize BM25 scores to [0, 1] range
        min_b = float(raw_scores.min()) if len(raw_scores) > 0 else 0.0
        max_b = float(raw_scores.max()) if len(raw_scores) > 0 else 1.0
        b_range = (max_b - min_b) if (max_b > min_b) else 1.0
        bm25_norm = [(float(s) - min_b) / b_range if max_b > min_b else 0.0 for s in raw_scores]

        # Dense scores are cosine similarities already bounded in [0, 1].
        # We blend dense cosine similarity directly with normalized BM25 scores
        # to avoid severe artificial compression of close similarity scores.
        for i, chunk in enumerate(candidates):
            dense_s = chunk["score"]
            chunk["bm25_score"]   = round(bm25_norm[i], 4)
            chunk["hybrid_score"] = round(
                alpha * dense_s + (1.0 - alpha) * bm25_norm[i], 4
            )

        # Strict filter: scope AND where_filter must both be respected
        filtered = [
            c for c in candidates
            if (scope is None or scope_allows(scope, c))
            and matches_where_filter(c, where_filter)
        ]
        filtered.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return filtered

    def _cross_encode(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-score `candidates` with the cross-encoder and sort descending."""
        candidates = [{**c} for c in candidates]
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

    # ---------------------------------------------------------------------- #
    # Public search methods                                                    #
    # ---------------------------------------------------------------------- #

    def dense_search(
        self,
        query: str,
        where_filter: Optional[dict] = None,
        scope: Optional[dict] = None,
    ) -> list[dict]:
        """Pure vector similarity search with near-duplicate suppression."""
        candidates = self._fetch_dense(query, max(self.top_k * 6, 30), where_filter)
        filtered = [
            c for c in candidates
            if (scope is None or scope_allows(scope, c))
            and matches_where_filter(c, where_filter)
        ]
        deduped = deduplicate_chunks(filtered)
        return deduped[: self.top_k]

    def hybrid_search(
        self,
        query: str,
        where_filter: Optional[dict] = None,
        alpha: float = 0.7,
        scope: Optional[dict] = None,
    ) -> list[dict]:
        """Dense + BM25 blend with normalization and deduplication."""
        candidates = self._fetch_dense(query, max(self.top_k * 8, 40), where_filter)
        if not candidates:
            return []
        blended = self._apply_bm25(query, candidates, alpha, scope, where_filter)
        deduped = deduplicate_chunks(blended)
        return deduped[: self.top_k]

    def rerank_search(
        self,
        query: str,
        where_filter: Optional[dict] = None,
        alpha: float = 0.7,
        scope: Optional[dict] = None,
    ) -> list[dict]:
        """
        Hybrid retrieval -> cross-encoder reranker.

        Fetches candidates via dense search, blends with BM25,
        re-scores with the cross-encoder, deduplicates, and returns top_k.
        """
        if self.reranker is None:
            logger.warning("rerank mode requested but no reranker loaded; falling back to hybrid")
            return self.hybrid_search(query, where_filter, alpha, scope)

        n_candidates = max(self.top_k * self.rerank_factor, 40)
        candidates = self._fetch_dense(query, n_candidates, where_filter)
        if not candidates:
            return []
        blended = self._apply_bm25(query, candidates, alpha, scope, where_filter)
        reranked = self._cross_encode(query, blended)
        deduped = deduplicate_chunks(reranked)
        return deduped[: self.top_k]

    # ---------------------------------------------------------------------- #
    # Alpha sweep (A/B helper)                                                #
    # ---------------------------------------------------------------------- #

    def retrieve_sweep(
        self,
        query: str,
        alphas: list[float],
        where_filter: Optional[dict] = None,
        scope: Optional[dict] = None,
        apply_reranker: bool = False,
    ) -> list[dict]:
        """
        Run hybrid retrieval at each alpha value and return a list of records:

            [{"alpha": 0.5, "chunks": [...], "latency_ms": 12}, ...]
        """
        results = []
        candidates_cache = None
        for alpha in alphas:
            t0 = time.perf_counter()

            if candidates_cache is None:
                candidates_cache = self._fetch_dense(query, max(self.top_k * 8, 40), where_filter)

            blended = self._apply_bm25(query, list(candidates_cache), alpha, scope, where_filter)
            if apply_reranker and self.reranker is not None:
                blended = self._cross_encode(query, blended)
            deduped = deduplicate_chunks(blended)
            final = deduped[: self.top_k]

            results.append({
                "alpha":      alpha,
                "chunks":     final,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            })
        return results

    # ---------------------------------------------------------------------- #
    # Main entry point                                                         #
    # ---------------------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        where_filter: Optional[dict] = None,
        scope: Optional[dict] = None,
        alpha: float = 0.7,
        enable_query_understanding: bool = True,
    ) -> list[dict]:
        """
        Main entry point.

        mode: "dense" | "hybrid" | "rerank"
        alpha: BM25 blend weight (hybrid / rerank only). Env HYBRID_ALPHA overrides.
        """
        alpha = float(os.getenv("HYBRID_ALPHA", str(alpha)))

        # Query understanding: infer entity or doc_type if not explicitly set
        if enable_query_understanding:
            inferred = understand_query(query)
            if inferred["company"] and (where_filter is None or "company" not in where_filter):
                if scope is None or (scope.get("type") != "company_restricted"):
                    wf = dict(where_filter) if where_filter else {}
                    wf["company"] = inferred["company"]
                    where_filter = wf
            elif inferred["doc_type"] and (where_filter is None or "doc_type" not in where_filter):
                wf = dict(where_filter) if where_filter else {}
                wf["doc_type"] = inferred["doc_type"]
                where_filter = wf

        if mode == "rerank":
            return self.rerank_search(query, where_filter, alpha, scope)
        if mode == "hybrid":
            return self.hybrid_search(query, where_filter, alpha, scope)
        return self.dense_search(query, where_filter, scope)
