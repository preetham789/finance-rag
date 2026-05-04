# scripts/inventory.py
"""
Prints a summary of everything in data/raw/.
Run this after collection to verify your corpus before Phase 2.

What a healthy corpus looks like:
  - At least 30 documents
  - Mix of sources (don't over-index one company)
  - Total size 50MB–500MB (anything smaller = too thin)
  - No zero-byte files (failed downloads masquerading as files)
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR


def format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 ** 2:
        return f"{bytes_ / 1024:.1f} KB"
    else:
        return f"{bytes_ / (1024**2):.1f} MB"


def main():
    print("=" * 55)
    print("Corpus Inventory")
    print(f"Root: {RAW_DIR}")
    print("=" * 55)

    all_files = []
    by_source = defaultdict(list)  # source_folder → list of files

    # Walk all subdirectories
    for subdir in sorted(RAW_DIR.iterdir()):
        if subdir.is_dir():
            pdfs = list(subdir.glob("*.pdf")) + list(subdir.glob("*.txt"))
            for f in pdfs:
                all_files.append(f)
                by_source[subdir.name].append(f)
        elif subdir.suffix.lower() in (".pdf", ".txt"):
            all_files.append(subdir)
            by_source["(root)"].append(subdir)

    if not all_files:
        print("\nNo documents found in data/raw/")
        print("Run the collector scripts first:")
        print("  python scripts\\collect_rbi.py")
        print("  python scripts\\collect_bse.py --guide")
        print("  python scripts\\collect_bse.py --organize")
        return

    # ── Per-source breakdown ──
    print()
    for source, files in by_source.items():
        total_size = sum(f.stat().st_size for f in files)
        zero_byte  = [f for f in files if f.stat().st_size == 0]
        print(f"  [{source}]")
        print(f"    Files      : {len(files)}")
        print(f"    Total size : {format_size(total_size)}")
        if zero_byte:
            print(f"    WARN: {len(zero_byte)} zero-byte file(s) — likely failed downloads:")
            for z in zero_byte:
                print(f"      - {z.name}")
        print()

    # ── Overall summary ──
    total_size = sum(f.stat().st_size for f in all_files)
    zero_byte_total = [f for f in all_files if f.stat().st_size == 0]

    print("=" * 55)
    print(f"  Total documents : {len(all_files)}")
    print(f"  Total size      : {format_size(total_size)}")
    print(f"  Zero-byte files : {len(zero_byte_total)}")
    print()

    # ── Readiness assessment ──
    issues = []
    if len(all_files) < 20:
        issues.append(f"Only {len(all_files)} documents — aim for at least 30")
    if total_size < 5 * 1024 * 1024:
        issues.append("Total corpus under 5MB — likely too thin for meaningful RAG")
    if len(zero_byte_total) > 0:
        issues.append(f"{len(zero_byte_total)} zero-byte files — delete and re-download")
    if len(by_source) < 2:
        issues.append("Only one source type — add RBI + BSE for diversity")

    if issues:
        print("Issues to fix before Phase 2:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Corpus looks healthy.")
        print("Ready for Phase 2 — PDF parsing and chunking.")
    print("=" * 55)


if __name__ == "__main__":
    main()