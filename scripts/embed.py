# scripts/embed.py
"""
Embeds all chunks into ChromaDB.

Usage:
  python scripts\embed.py                         # embed chunks_recursive.jsonl
  python scripts\embed.py --verify-only           # just run test queries
  python scripts\embed.py --strategy sentence     # embed a different strategy file

Expected runtime on your machine:
  33,229 chunks / ~300 chunks per second ≈ ~2 minutes
  (First run downloads the model ~90MB — adds ~30 seconds)
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, CHROMA_DIR
from src.ingestion.embedder import (
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    BATCH_SIZE,
    load_chunks_from_jsonl,
    build_chroma_client,
    get_or_create_collection,
    embed_and_store,
    verify_collection,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["recursive", "token", "sentence"],
        help="Which chunks file to embed (default: recursive)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip embedding, just run verification queries",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Chunks per embedding batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 — Embedding Pipeline")
    print("=" * 60)

    # ── Step 1: Connect to ChromaDB ──
    print(f"\nStep 1: Connecting to ChromaDB at {CHROMA_DIR}")
    client     = build_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    print(f"  Collection '{COLLECTION_NAME}': {collection.count()} vectors currently stored")

    if args.verify_only:
        verify_collection(collection)
        return

    # ── Step 2: Load chunks ──
    chunks_path = PROCESSED_DIR / f"chunks_{args.strategy}.jsonl"
    if not chunks_path.exists():
        print(f"\nERROR: {chunks_path} not found.")
        print(f"Run first: python scripts\\ingest.py --strategy {args.strategy}")
        return

    print(f"\nStep 2: Loading chunks from {chunks_path.name}")
    chunks = load_chunks_from_jsonl(chunks_path)
    print(f"  Loaded {len(chunks):,} chunks")

    # ── Step 3: Load embedding model ──
    print(f"\nStep 3: Loading embedding model '{EMBEDDING_MODEL_NAME}'")
    print(f"  First run downloads ~90MB to ~/.cache/huggingface/")
    print(f"  Subsequent runs load from cache (fast)\n")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Print model info
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model loaded. Embedding dimension: {dim}")
    print(f"  Batch size: {args.batch_size}")
    estimated_minutes = len(chunks) / (300 * 60)
    print(f"  Estimated time: ~{estimated_minutes:.1f} minutes\n")

    # ── Step 4: Embed and store ──
    print(f"Step 4: Embedding {len(chunks):,} chunks into ChromaDB...")
    stats = embed_and_store(
        chunks=chunks,
        collection=collection,
        model=model,
        batch_size=args.batch_size,
    )

    # ── Step 5: Results ──
    print(f"\n{'='*60}")
    print(f"Embedding complete")
    print(f"  Chunks embedded  : {stats['embedded']:,}")
    print(f"  Errors           : {stats['errors']}")
    print(f"  Time taken       : {stats['elapsed_seconds']}s")
    print(f"  Speed            : {stats['chunks_per_second']} chunks/sec")
    print(f"  Total in store   : {collection.count():,} vectors")
    print(f"  ChromaDB location: {CHROMA_DIR}")

    # ── Step 6: Verify ──
    verify_collection(collection)

    print(f"\n{'='*60}")
    print("Phase 3 complete.")
    print("Next: Phase 4 — build the RAG query chain")
    print("  python scripts\\query.py --question \"What was RBI's inflation target in 2024?\"")


if __name__ == "__main__":
    main()