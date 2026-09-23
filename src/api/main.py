# src/api/main.py
"""
FastAPI backend for the Finance RAG system.

Endpoints:
  GET  /health          — liveness check
  GET  /stats           — corpus statistics
  POST /query           — ask a question, get a grounded answer
  POST /query/stream    — streaming version (Phase 6 bonus)

Why FastAPI over Flask?
  - Async by default — handles concurrent requests without blocking
  - Auto-generates OpenAPI docs at /docs — recruiters can try your API
  - Pydantic validation — request/response schemas are self-documenting
  - Industry standard for ML serving in 2026
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import CHROMA_DIR
from src.ingestion.embedder import (
    EMBEDDING_MODEL_NAME, COLLECTION_NAME,
    build_chroma_client, get_or_create_collection,
)
from src.retrieval.retriever import FinanceRetriever
from src.generation.chain import FinanceRAGChain
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──
rag_chain: Optional[FinanceRAGChain] = None
vector_count: int = 0
_retriever: Optional[FinanceRetriever] = None   # BUG 7 FIX: kept so top_k can be overridden per-request


# BUG 5 FIX: Replaced deprecated @app.on_event("startup") with lifespan context manager
# (required since FastAPI 0.93 — generates deprecation warnings on 0.115.4)
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

    groq_key   = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if groq_key:
        rag_chain = FinanceRAGChain(_retriever, api_key=groq_key, provider="groq")
        logger.info(f"RAG system ready — {vector_count:,} vectors, Groq LLM")
    elif openai_key:
        rag_chain = FinanceRAGChain(_retriever, api_key=openai_key, provider="openai")
        logger.info(f"RAG system ready — {vector_count:,} vectors, OpenAI LLM")
    else:
        logger.error("No API key found — LLM will not work")

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
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request / Response schemas ──

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500,
                          example="What was RBI's repo rate in 2024?")
    company:  Optional[str] = Field(None, example="HDFC Bank")
    doc_type: Optional[str] = Field(None, example="rbi")
    top_k:    int           = Field(5, ge=1, le=10)

class SourceInfo(BaseModel):
    company:     str
    page_number: int
    source_file: str
    score:       float
    preview:     str

class QueryResponse(BaseModel):
    question:       str
    answer:         str
    sources:        list[SourceInfo]
    retrieval_mode: str
    model:          str
    tokens_used:    int
    latency_ms:     int


# ── Endpoints ──

@app.get("/health")
async def health():
    """Liveness check — used by Docker and load balancers."""
    return {
        "status":       "ok",
        "vectors":      vector_count,
        "model_loaded": rag_chain is not None,
    }


@app.get("/stats")
async def stats():
    """Corpus statistics."""
    return {
        "total_vectors":   vector_count,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection":      COLLECTION_NAME,
        "chroma_dir":      str(CHROMA_DIR),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question. Returns a grounded answer with source citations.

    Optionally filter by company or doc_type to narrow the search.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not loaded")

    # BUG 7 FIX: Apply per-request top_k to the retriever before each query
    if _retriever is not None:
        _retriever.top_k = request.top_k

    # Build where filter
    where_filter = None
    if request.company:
        where_filter = {"company": request.company}
    elif request.doc_type:
        where_filter = {"doc_type": request.doc_type}

    start = time.time()
    try:
        result = rag_chain.query(
            question=request.question,
            where_filter=where_filter,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - start) * 1000)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        retrieval_mode=result["retrieval_mode"],
        model=result["model"],
        tokens_used=result.get("tokens_used", 0),
        latency_ms=latency_ms,
    )