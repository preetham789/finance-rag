# src/ingestion/embedder.py
"""
Embedder — converts chunks into vectors and stores them in ChromaDB.

Key decisions made here (and why):

  Embedding model: BAAI/bge-small-en-v1.5 (local, free, fast)
    - Runs entirely on your machine — no API calls, no cost, no rate limits
    - 384-dimensional vectors (vs 1536 for OpenAI) — 4x smaller, 4x faster
    - Scores within 5% of OpenAI on financial text retrieval benchmarks
    - Industry uses this exact model for cost-sensitive production RAG

  Why not OpenAI embeddings yet?
    - Each embedding call costs money. 33,229 chunks × ~150 tokens = ~5M tokens
    - At $0.02/1M tokens that's ~$0.10 — cheap, but teaches bad habits
    - Learn local first, then swap to cloud in Phase 6 when you understand tradeoffs

  ChromaDB over FAISS:
    - ChromaDB persists to disk automatically — restart your machine, data is safe
    - Built-in metadata filtering: filter by company="HDFC Bank" at query time
    - FAISS is faster at huge scale (millions of vectors) — overkill for 33k chunks
"""

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

TELEMETRY_LOGGERS = (
    "chromadb.telemetry.product.posthog",
    "posthog",
)

# ── Embedding model — downloaded once, cached in ~/.cache/huggingface/ ──
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ── ChromaDB collection name ──
COLLECTION_NAME = "finance_rag"

# ── How many chunks to embed in one batch ──
# 64 is safe for 8GB RAM. Increase to 128 if you have 16GB.
BATCH_SIZE = 64


def load_chunks_from_jsonl(jsonl_path: Path) -> list[dict]:
    """
    Load chunks from the JSONL file produced by scripts/ingest.py.
    Skips the metadata header line (first line with _type=run_metadata).
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_type") == "run_metadata":
                continue   # skip header
            chunks.append(obj)

    logger.info(f"Loaded {len(chunks)} chunks from {jsonl_path.name}")
    return chunks


def build_chroma_client(persist_dir: Path) -> chromadb.PersistentClient:
    """
    Create a ChromaDB client that persists to disk.

    PersistentClient saves everything to persist_dir automatically.
    On restart, just point to the same directory — your vectors are still there.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    # Chroma telemetry can still log compatibility errors even when disabled.
    for logger_name in TELEMETRY_LOGGERS:
        logging.getLogger(logger_name).disabled = True
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get the collection if it exists, create it if not.

    Why cosine distance?
      Embeddings are unit-normalized vectors. Cosine similarity measures the
      angle between two vectors — 1.0 means identical meaning, 0.0 means
      unrelated. It's the standard for semantic search because it ignores
      vector magnitude (document length) and focuses purely on direction (meaning).
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for semantic search
    )
    return collection


def chunk_batches(items: list, batch_size: int) -> Iterator[list]:
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Embed all chunks in batches and upsert into ChromaDB.

    Uses upsert (not add) so re-running this script is safe —
    existing chunks get updated rather than duplicated.

    What ChromaDB stores per chunk:
      - id        : unique chunk_id string (e.g. "RBI_p4_c0_rec")
      - embedding : list of 384 floats — the semantic fingerprint
      - document  : the raw chunk text — returned in query results
      - metadata  : source_file, company, page_number, doc_type, etc.

    The metadata is what enables filtered retrieval:
      collection.query(where={"company": "HDFC Bank"})
    """
    # Check how many are already in the collection
    existing_count = collection.count()
    if existing_count > 0:
        print(f"  Collection already has {existing_count} vectors.")
        print(f"  Using upsert — will update existing, add new.\n")

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    stats = {"embedded": 0, "errors": 0}
    start_time = time.time()

    for batch in tqdm(
        chunk_batches(chunks, batch_size),
        total=total_batches,
        desc="Embedding batches",
        unit="batch",
    ):
        try:
            # ── Extract fields for this batch ──
            ids        = [c["chunk_id"] for c in batch]
            texts      = [c["text"] for c in batch]
            metadatas  = [
                {
                    # ChromaDB metadata values must be str, int, float, or bool
                    # Convert anything else to string
                    "source_file":  str(c["source_file"]),
                    "page_number":  int(c["page_number"]),
                    "doc_type":     str(c["doc_type"]),
                    "company":      str(c["company"]),
                    "strategy":     str(c["strategy"]),
                    "word_count":   int(c["word_count"]),
                    "chunk_index":  int(c["chunk_index"]),
                }
                for c in batch
            ]

            # ── Embed — this is the expensive step ──
            # BGE models work best with a query prefix for retrieval tasks.
            # For documents being stored, no prefix needed.
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # unit normalize for cosine similarity
            ).tolist()

            # ── Upsert into ChromaDB ──
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            stats["embedded"] += len(batch)

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            stats["errors"] += len(batch)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed)
    stats["chunks_per_second"] = round(len(chunks) / elapsed, 1)
    return stats


def verify_collection(collection: chromadb.Collection) -> None:
    """
    Run a quick sanity check query to confirm the vector store works.
    Tests both an RBI query and a company query.
    """
    print("\nVerifying collection with test queries...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    test_queries = [
        ("RBI inflation target", {"doc_type": "rbi"}),
        ("HDFC Bank NPA ratio", {"company": "HDFC Bank"}),
        ("TCS revenue growth", {"company": "TCS"}),
    ]

    for query_text, where_filter in test_queries:
        query_embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        print(f"\n  Query: '{query_text}'  (filter: {where_filter})")
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                score = round(1 - dist, 3)  # cosine distance → similarity score
                print(f"    [{i+1}] score={score}  company={meta['company']}  "
                      f"page={meta['page_number']}")
                print(f"         {doc[:120].strip()}...")
        else:
            print("    No results found — check filter values")
