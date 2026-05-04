# scripts/debug_retrieval.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DIR

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection("finance_rag")
print(f"Vectors in store: {collection.count()}\n")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed(text: str) -> list:
    """
    Returns a flat list of floats — exactly what ChromaDB expects.
    The bug was: model.encode([text]).tolist() → shape (1,384) → [[...]]
    ChromaDB query_embeddings wants [[...]] but was getting [[[...]]]
    Fix: encode a plain string (not a list) → shape (384,) → [...]
    Then wrap in a list at query time: query_embeddings=[[...]]
    """
    vec = model.encode(
        text,                       # ← plain string, not [text]
        normalize_embeddings=True,
    )
    return vec.tolist()             # → flat list of 384 floats

def debug_query(query_text: str, where_filter: dict = None, n: int = 4):
    print("=" * 65)
    print(f"QUERY: '{query_text}'")
    if where_filter:
        print(f"FILTER: {where_filter}")
    print("=" * 65)

    prefixed = f"Represent this query for searching relevant passages: {query_text}"
    embedding = embed(prefixed)     # flat list [float, float, ...]

    kwargs = {
        "query_embeddings": [embedding],   # ChromaDB wants [[...]]
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        print(f"Query failed: {e}\n")
        return

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        print("No results returned.\n")
        return

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        score = round(1 - dist, 4)
        print(f"\n[Chunk {i}]  Company={meta.get('company')}  "
              f"Page={meta.get('page_number')}  Score={score}")
        print("-" * 55)
        print(doc)    # full text — no truncation
    print()


# ── Run diagnostics ──

debug_query(
    "HDFC Bank gross NPA non performing assets ratio",
    where_filter={"company": "HDFC Bank"},
    n=4,
)

debug_query(
    "repo rate monetary policy committee decision 2024",
    where_filter={"doc_type": "rbi"},
    n=3,
)