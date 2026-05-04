# scripts/ingest.py
"""
Full ingestion pipeline — run this to process all 53 PDFs.

Usage:
  python scripts\ingest.py                    # process all, recursive strategy
  python scripts\ingest.py --strategy sentence
  python scripts\ingest.py --strategy token
  python scripts\ingest.py --compare         # compare all 3 strategies, don't save
  python scripts\ingest.py --limit 5         # process first 5 PDFs only (fast test)

Output:
  data/processed/chunks_<strategy>.jsonl     # one chunk per line, JSON format

JSONL (JSON Lines) format: one JSON object per line.
Why JSONL over CSV?
  Each chunk has variable-length text. CSV struggles with multi-line text
  and special characters. JSONL is simple, appendable, and universally
  readable by pandas, HuggingFace, and every ML tool.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.loader import load_all_documents
from src.ingestion.chunker import page_dicts_to_chunks, compare_strategies

# ── Logging setup ──
# INFO level shows progress. Set to DEBUG to see every skipped page.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def save_chunks_jsonl(chunks: list[dict], output_path: Path) -> None:
    """
    Save chunks as JSONL — one JSON object per line.
    Appends a run metadata header as the first line.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Line 1: run metadata (useful for debugging old experiment outputs)
        metadata = {
            "_type": "run_metadata",
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "strategy": chunks[0]["strategy"] if chunks else "unknown",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        # Remaining lines: one chunk per line
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(chunks)} chunks → {output_path}  ({size_mb:.1f} MB)")


def print_sample_chunks(chunks: list[dict], n: int = 3) -> None:
    """
    Print n sample chunks so you can visually verify quality.
    Always shows one RBI chunk and one company chunk if available.
    """
    print(f"\n{'='*60}")
    print("Sample chunks (visual quality check)")
    print("="*60)

    rbi_chunks     = [c for c in chunks if c["doc_type"] == "rbi"]
    company_chunks = [c for c in chunks if c["doc_type"] == "company"]

    samples = []
    if rbi_chunks:     samples.append(rbi_chunks[0])
    if company_chunks: samples.append(company_chunks[0])
    # Add a chunk from the middle of the list for variety
    if len(chunks) > 100:
        samples.append(chunks[len(chunks) // 2])

    for i, chunk in enumerate(samples[:n], 1):
        print(f"\n--- Sample {i} ---")
        print(f"  ID       : {chunk['chunk_id']}")
        print(f"  Source   : {chunk['source_file'][:50]}")
        print(f"  Company  : {chunk['company']}")
        print(f"  Page     : {chunk['page_number']}")
        print(f"  Words    : {chunk['word_count']}")
        print(f"  Text     :\n")
        # Wrap text at 70 chars for readability in terminal
        words = chunk["text"].split()
        line, lines_out = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 70:
                lines_out.append("    " + " ".join(line))
                line = []
        if line:
            lines_out.append("    " + " ".join(line))
        print("\n".join(lines_out[:8]))  # show first 8 lines max
        if len(lines_out) > 8:
            print(f"    ... ({chunk['word_count'] - 60} more words)")

    print(f"\n{'='*60}")


def print_corpus_stats(chunks: list[dict]) -> None:
    """Print a breakdown of chunks by company and doc type."""
    from collections import Counter

    print(f"\n{'='*60}")
    print("Corpus breakdown by company")
    print("="*60)

    by_company = Counter(c["company"] for c in chunks)
    for company, count in by_company.most_common():
        bar = "█" * (count // 50)
        print(f"  {company:<25} {count:>5} chunks  {bar}")

    by_type = Counter(c["doc_type"] for c in chunks)
    print(f"\nBy type:")
    for doc_type, count in by_type.items():
        print(f"  {doc_type:<12} {count:>5} chunks")

    word_counts = [c["word_count"] for c in chunks]
    print(f"\nWord count stats:")
    print(f"  Total words   : {sum(word_counts):,}")
    print(f"  Avg per chunk : {sum(word_counts)//len(word_counts)}")
    print(f"  Min           : {min(word_counts)}")
    print(f"  Max           : {max(word_counts)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "token", "sentence"],
        default="recursive",
        help="Chunking strategy to use (default: recursive)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all 3 strategies on a sample — don't save output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N PDFs (for quick testing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Ingestion Pipeline — Phase 2")
    print("=" * 60)

    # ── Step 1: Load all PDFs ──
    print("\nStep 1: Loading and cleaning PDFs...")
    pages = load_all_documents(RAW_DIR)
    print(f"\nLoaded {len(pages)} clean pages total")

    # Apply limit if set (for quick testing)
    if args.limit:
        # Get pages from first N unique files
        seen_files, limited_pages = set(), []
        for p in pages:
            seen_files.add(p["source_file"])
            if len(seen_files) <= args.limit:
                limited_pages.append(p)
        pages = limited_pages
        print(f"Limited to first {args.limit} files → {len(pages)} pages")

    if not pages:
        print("ERROR: No pages extracted. Check your PDFs in data/raw/")
        return

    # ── Step 2: Compare or chunk ──
    if args.compare:
        print("\nStep 2: Comparing chunking strategies...")
        compare_strategies(pages, sample_pages=20)
        return

    print(f"\nStep 2: Chunking with strategy='{args.strategy}'...")
    chunks = page_dicts_to_chunks(
        pages,
        strategy=args.strategy,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"Produced {len(chunks)} chunks")

    # ── Step 3: Quality checks ──
    print("\nStep 3: Quality checks...")
    print_sample_chunks(chunks)
    print_corpus_stats(chunks)

    # ── Step 4: Save ──
    print("\nStep 4: Saving...")
    output_path = PROCESSED_DIR / f"chunks_{args.strategy}.jsonl"
    save_chunks_jsonl(chunks, output_path)

    print("\nPhase 2 complete.")
    print(f"Next: run Phase 3 — embed chunks into ChromaDB")
    print(f"  python scripts\\ingest.py --strategy recursive  (already done)")
    print(f"  Then: python scripts\\embed.py")


if __name__ == "__main__":
    main()