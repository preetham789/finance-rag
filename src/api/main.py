# src/api/main.py
"""
FastAPI backend for the Finance RAG system.

Endpoints:
  GET  /health          – liveness check
  GET  /stats           – corpus statistics
  POST /query           – ask a question, get a grounded answer
  POST /query/stream    – streaming version (Phase 6 bonus)

Why FastAPI over Flask?
  - Async by default – handles concurrent requests without blocking
  - Auto-generates OpenAPI docs at /docs – recruiters can try your API
  - Pydantic validation – request/response schemas are self-documenting
  - Industry standard for ML serving in 2026
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DIR
from src.generation.chain import FinanceRAGChain, LLMGenerationError
from src.ingestion.embedder import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    build_chroma_client,
    get_or_create_collection,
)
from src.retrieval.retriever import FinanceRetriever
from src.security import (
    Principal,
    build_scope,
    enforce_rate_limit,
    scope_to_chroma_where,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──
rag_chain: Optional[FinanceRAGChain] = None
vector_count: int = 0
_retriever: Optional[FinanceRetriever] = None
_RESPONSE_CACHE: dict[str, dict] = {}
_MAX_CACHE_SIZE = 500


# BUG 5 FIX: Replaced deprecated @app.on_event("startup") with lifespan context manager
# (required since FastAPI 0.93 – generates deprecation warnings on 0.115.4)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to ChromaDB when the server starts."""
    global rag_chain, vector_count, _retriever
    logger.info("Loading RAG system...")

    client     = build_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    vector_count = collection.count()

    model      = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # BUG 7 FIX: Store retriever separately so per-request top_k can be applied
    _retriever = FinanceRetriever(collection, model, top_k=5)

    # Optional cross-encoder reranker – loaded only when RERANKER_MODEL is set.
    reranker_model = os.getenv("RERANKER_MODEL", "")
    if reranker_model:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            _retriever.reranker = CrossEncoder(reranker_model)
            logger.info("Cross-encoder reranker loaded: %s", reranker_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load reranker %s: %s", reranker_model, exc)

    groq_key   = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if groq_key:
        rag_chain = FinanceRAGChain(_retriever, api_key=groq_key, provider="groq")
        logger.info("RAG system ready – %d vectors, Groq LLM", vector_count)
    elif openai_key:
        rag_chain = FinanceRAGChain(_retriever, api_key=openai_key, provider="openai")
        logger.info("RAG system ready – %d vectors, OpenAI LLM", vector_count)
    else:
        logger.error("No API key found – LLM will not work")

    yield  # server is running

    # Shutdown: nothing to clean up for ChromaDB PersistentClient
    logger.info("Shutting down RAG system.")


# ── FastAPI app ──
app = FastAPI(
    title="Finance RAG API",
    description="Q&A over Indian company annual reports and RBI publications",
    version="1.0.0",
    lifespan=lifespan,
)

# BUG 6 FIX: Restrict CORS to known origins instead of wildcard "*"
# For local dev both Streamlit (8501) and API docs (8000) are allowed.
# In production, replace with your actual deployed frontend URL.
_ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# ── Request / Response schemas ──

class QueryRequest(BaseModel):
    question:       str           = Field(..., min_length=5, max_length=500,
                                          example="What was RBI's repo rate in 2024?")
    company:        Optional[str] = Field(None, example="HDFC Bank")
    doc_type:       Optional[str] = Field(None, example="rbi")
    top_k:          int           = Field(5, ge=1, le=10)
    retrieval_mode: str           = Field(
        "hybrid",
        pattern="^(dense|hybrid|rerank)$",
        description="Retrieval strategy: dense | hybrid | rerank",
    )
    alpha:          float         = Field(
        0.7, ge=0.0, le=1.0,
        description="BM25 blend weight (hybrid / rerank only). 1.0 = pure dense.",
    )


class SourceInfo(BaseModel):
    company:     str
    page_number: int
    source_file: str
    score:       float
    preview:     str
    chunk_id:    Optional[str] = None


class QueryResponse(BaseModel):
    question:       str
    answer:         str
    sources:        list[SourceInfo]
    retrieval_mode: str
    model:          str
    tokens_used:    int
    latency_ms:     int
    grounding:      Optional[dict] = None
    timings:        Optional[dict] = None


# ── Endpoints ──

@app.get("/health")
def health():
    """Liveness & readiness check – verifies vector store and model status."""
    is_ready = rag_chain is not None and vector_count > 0
    return {
        "status":       "ok" if is_ready else "degraded",
        "ready":        is_ready,
        "vectors":      vector_count,
        "model_loaded": rag_chain is not None,
    }


@app.get("/stats")
def stats():
    """Corpus statistics."""
    return {
        "total_vectors":   vector_count,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection":      COLLECTION_NAME,
        "chroma_dir":      str(CHROMA_DIR),
    }


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    principal: Principal = Depends(enforce_rate_limit),
):
    """
    Ask a question. Returns a grounded answer with source citations.

    Executed in FastAPI's threadpool ('def' rather than 'async def') to prevent
    blocking the asyncio event loop on synchronous PyTorch/Chroma operations.
    Optionally filter by company or doc_type to narrow the search.
    Requires API-key authentication (unless AUTH_MODE=off).
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")

    # In-memory query cache check
    cache_key = (
        f"{request.question}:{request.company}:{request.doc_type}:"
        f"{request.retrieval_mode}:{request.alpha}:{request.top_k}:"
        f"{getattr(principal, 'role', '')}"
    )
    if cache_key in _RESPONSE_CACHE:
        logger.info("Serving query from cache for question: %s", request.question[:50])
        return QueryResponse(**_RESPONSE_CACHE[cache_key])

    # BUG 7 FIX: Apply per-request top_k to the retriever before each query
    if _retriever is not None:
        _retriever.top_k = request.top_k

    # Access control: build scope and fail-closed Chroma where clause
    scope = build_scope(principal, company=request.company, doc_type=request.doc_type)
    where_filter = scope_to_chroma_where(scope)

    start = time.time()
    try:
        result = rag_chain.query(
            question=request.question,
            retrieval_mode=request.retrieval_mode,
            where_filter=where_filter,
            scope=scope,
            alpha=request.alpha,
            key_id=getattr(principal, "key_id", None),
        )
    except HTTPException:
        raise
    except LLMGenerationError as e:
        logger.error("LLM generation error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)

    tokens_used = result.get("tokens_used", 0)
    if not isinstance(tokens_used, int):
        try:
            tokens_used = int(tokens_used)
        except (ValueError, TypeError):
            tokens_used = 0

    # Structured per-request log (supplements chain-level log with principal info)
    grounding = result.get("grounding") or {}
    logger.info(
        "api_request %s",
        json.dumps({
            "key_id":          getattr(principal, "key_id", None),
            "retrieval_mode":  request.retrieval_mode,
            "alpha":           request.alpha,
            "tokens":          tokens_used,
            "latency_ms":      latency_ms,
            "timings":         result.get("timings"),
            "grounding_ok":    grounding.get("ok"),
        }),
    )

    response_payload = {
        "question":       result["question"],
        "answer":         result["answer"],
        "sources":        [SourceInfo(**s).model_dump() for s in result["sources"]],
        "retrieval_mode": result["retrieval_mode"],
        "model":          result["model"],
        "tokens_used":    tokens_used,
        "latency_ms":     latency_ms,
        "grounding":      result.get("grounding"),
        "timings":        result.get("timings"),
    }

    # Store in cache
    if len(_RESPONSE_CACHE) >= _MAX_CACHE_SIZE:
        _RESPONSE_CACHE.pop(next(iter(_RESPONSE_CACHE)))
    _RESPONSE_CACHE[cache_key] = response_payload

    return QueryResponse(**response_payload)
