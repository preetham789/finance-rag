# scripts/deduplicate_rbi.py
"""
Removes duplicate RBI full annual report files.
Keeps only the chapter-specific files and one copy of the full report.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR

rbi_dir = RAW_DIR / "rbi"

# These are the duplicate full-report copies — keep only RBI_AR_ version
duplicates = [
    "RBI_FSR_Annual_Report_2024-25.pdf",
    "RBI_MPR_Annual_Report_2024-25.pdf",
    "RBI_RCF_Annual_Report_2024-25.pdf",
]

for fname in duplicates:
    path = rbi_dir / fname
    if path.exists():
        size_mb = path.stat().st_size / (1024*1024)
        path.unlink()
        print(f"  Deleted: {fname}  ({size_mb:.1f} MB freed)")
    else:
        print(f"  Not found: {fname}")

# Also check for 0ANNUALREPORT202425 which duplicates RBI_AR_Annual_Report_2024-25
# Compare page counts to confirm before deleting
candidates = [
    "0ANNUALREPORT202425DA4AE08189C848C8846718B080F2A0A9.pdf",
    "0ANNUALREPORT202324_FULLDF549205FA214F62A2441C5320D64A29.pdf",
]
print("\nCandidate duplicates (check manually before deleting):")
for fname in candidates:
    path = rbi_dir / fname
    if path.exists():
        size_mb = path.stat().st_size / (1024*1024)
        print(f"  {fname[:50]}  ({size_mb:.1f} MB)")

print("\nDone. Re-run: python scripts\\inventory.py to verify.")