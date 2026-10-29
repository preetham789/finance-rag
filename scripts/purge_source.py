#!/usr/bin/env python
"""Inspect or delete vectors by source_file, e.g. after replacing a wrong PDF.

Replacing the PDF on disk does nothing to ChromaDB: the old chunks stay indexed, and
upsert only overwrites matching IDs, so a different file leaves them behind.

  python scripts/purge_source.py --list HDFC                 # which sources match
  python scripts/purge_source.py --source "<source_file>"    # dry run: counts only
  python scripts/purge_source.py --source "<source_file>" --yes

Then re-run ingest.py and embed.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

COLLECTION = "finance_rag"


def get_collection(path: Optional[str] = None):
    import chromadb
    from chromadb.config import Settings

    if path is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from src.config import CHROMA_DIR  # type: ignore

            path = str(CHROMA_DIR)
        except Exception:
            path = os.getenv("CHROMA_PERSIST_DIR", "experiments/chromadb")
    client = chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(COLLECTION)


def list_sources(col, needle: str) -> Counter:
    needle = needle.lower()
    counts: Counter = Counter()
    for meta in col.get(include=["metadatas"])["metadatas"]:
        company, source = meta.get("company", ""), meta.get("source_file", "")
        if needle in str(company).lower() or needle in str(source).lower():
            counts[(company, source)] += 1
    return counts


def purge(col, source_file: str, *, apply: bool, batch: int = 5000) -> tuple[int, int]:
    ids = col.get(where={"source_file": source_file}, include=[])["ids"]
    if apply:
        for i in range(0, len(ids), batch):
            col.delete(ids=ids[i : i + batch])
    return len(ids), col.count()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", metavar="TEXT", help="list company/source_file matching TEXT")
    ap.add_argument("--source", help="exact source_file metadata value to delete")
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--chroma-dir", default=None)
    args = ap.parse_args(argv)

    col = get_collection(args.chroma_dir)
    print(f"collection '{COLLECTION}': {col.count()} vectors")
    if args.list:
        for (company, source), n in sorted(list_sources(col, args.list).items()):
            print(f"{n:>6}  company={company!r}  source_file={source!r}")
        return 0
    if not args.source:
        ap.error("give --list TEXT or --source FILE")
    found, _ = purge(col, args.source, apply=False)
    if found == 0:
        print("no vectors with that source_file; use --list to find the exact value")
        return 1
    if not args.yes:
        print(f"dry run: would delete {found} vectors (add --yes to apply)")
        return 0
    found, remaining = purge(col, args.source, apply=True)
    print(f"deleted {found} vectors; {remaining} remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
