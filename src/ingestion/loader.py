# src/ingestion/loader.py
"""
PDF Loader — converts raw PDFs into clean, metadata-tagged documents.

Handles two document families discovered in our corpus:
  1. RBI reports  — clean text, repeating headers/footers to strip
  2. Company ARs  — cover pages with vertical design text to detect + skip

Output: list of dicts, one per page, each containing:
  {
    "text":        str,   # clean page text
    "source_file": str,   # original filename
    "page_number": int,   # 1-indexed
    "doc_type":    str,   # "rbi" or "company"
    "company":     str,   # e.g. "TCS" or "RBI"
    "word_count":  int,   # number of words on this page
  }
"""

import re
import logging
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Patterns that indicate a line is pure noise ──
# Compiled once at module load for speed.
NOISE_PATTERNS = [
    # Standalone page numbers:  "17", "18", " 42 "
    re.compile(r"^\s*\d{1,3}\s*$"),

    # RBI report headers on their own line
    re.compile(r"^\s*(ANNUAL REPORT|MONETARY POLICY REPORT|FINANCIAL STABILITY REPORT"
               r"|REPORT ON CURRENCY AND FINANCE|RBI BULLETIN"
               r"|RESERVE BANK OF INDIA)\s*(\d{4}[-–]\d{2,4})?\s*$", re.IGNORECASE),

    # Chapter headers repeated as running headers  e.g. "ECONOMIC REVIEW"
    re.compile(r"^\s*(ECONOMIC REVIEW|MONETARY POLICY OPERATIONS|CREDIT DELIVERY"
               r"|FINANCIAL MARKETS|REGULATION|PUBLIC DEBT|CURRENCY MANAGEMENT"
               r"|PAYMENT AND SETTLEMENT|GOVERNANCE|ACCOUNTS)\s*$", re.IGNORECASE),

    # BSE/NSE boilerplate footer lines
    re.compile(r"^\s*(BSE Limited|National Stock Exchange|CIN:|ISIN:)\s*", re.IGNORECASE),

    # Purely decorative lines: "---", "===", "***", "..."
    re.compile(r"^\s*[-=*_.]{3,}\s*$"),

    # Page reference lines: "See page 42", "Refer page 18"
    re.compile(r"^\s*(see|refer|continued on|contd\.?)\s+(page|pg\.?)\s*\d+\s*$",
               re.IGNORECASE),
]


def is_noise_line(line: str) -> bool:
    """Return True if this line is header/footer/decoration noise."""
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_vertical_design_text(text: str) -> bool:
    """
    Detect the vertical character-by-character design text found on
    BSE company AR cover pages (e.g. 'T\nR\nA\nN\nS\nF\nO\nR\nM\nI\nN\nG').

    Heuristic: if more than 60% of 'words' are single characters,
    and total word count is > 8, it's a design element, not real text.
    """
   # NEW — also catches post-cleaning collapsed vertical text
def is_vertical_design_text(text: str) -> bool:
    """
    Detect vertical character-by-character design text.
    Handles both raw form ('T\nR\nA\nN\nS') and
    collapsed form ('T R A N S F O R M I N G') after cleaning.
    """
    words = text.split()
    if len(words) < 8:
        return False

    single_char_count = sum(1 for w in words if len(w) == 1)
    ratio = single_char_count / len(words)

    # Raw form: >60% single chars
    if ratio > 0.60:
        return True

    # Collapsed form: look for runs of 5+ consecutive single-char words
    # e.g. "T R A N S F O R M I N G" — dead giveaway of design text
    consecutive = 0
    max_consecutive = 0
    for w in words:
        if len(w) == 1 and w.isalpha():
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    return max_consecutive >= 5


def clean_page_text(raw_text: str) -> str:
    """
    Clean a single page's extracted text.

    Steps:
      1. Split into lines
      2. Drop noise lines (headers, page numbers, decorations)
      3. Normalize whitespace within each line
      4. Collapse runs of blank lines to a single blank line
      5. Strip leading/trailing whitespace
    """
    lines = raw_text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Normalize internal whitespace (tabs → space, multi-space → single)
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            cleaned_lines.append("")   # preserve paragraph breaks
            continue

        if is_noise_line(line):
            continue

        cleaned_lines.append(line)

    # Collapse multiple consecutive blank lines → one blank line
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))

    return result.strip()


def extract_tables_as_text(page: fitz.Page) -> str:
    """
    Extract tables from a PDF page as labeled key-value text.

    Problem: PyMuPDF's get_text("text") extracts tables row by row,
    losing the column headers. A table like:

      Metric          FY2024    FY2023
      Gross NPA (%)    1.24      1.34
      Net NPA (%)      0.33      0.35

    becomes: "Gross NPA (%) 1.24 1.34 Net NPA (%) 0.33 0.35"
    - which the LLM can't interpret because headers are detached.

    Fix: use get_text("dict") which gives us block-level structure,
    then reconstruct tables with their headers intact.
    """
    blocks = page.get_text("dict")["blocks"]
    table_texts = []

    for block in blocks:
        if block.get("type") != 0:   # type 0 = text block
            continue

        lines = block.get("lines", [])
        if len(lines) < 2:
            continue

        line_texts = []
        for line in lines:
            spans = line.get("spans", [])
            line_text = " ".join(
                span["text"].strip()
                for span in spans
                if span["text"].strip()
            )
            if line_text:
                line_texts.append(line_text)

        avg_len = sum(len(line) for line in line_texts) / max(len(line_texts), 1)
        if len(line_texts) >= 3 and avg_len < 60:
            header = line_texts[0]
            for data_line in line_texts[1:]:
                if any(char.isdigit() for char in data_line):
                    table_texts.append(f"{header}: {data_line}")

    return "\n".join(table_texts)


def infer_doc_type(filepath: Path) -> tuple[str, str]:
    """
    Infer whether a PDF is an RBI document or a company annual report,
    and extract the company/entity name.

    Returns: (doc_type, entity_name)
      e.g. ("rbi", "RBI") or ("company", "TCS")
    """
    name_lower = filepath.stem.lower()

    # RBI documents live in the rbi/ subfolder
    if "rbi" in filepath.parts or "rbi" in name_lower:
        return "rbi", "RBI"

    # Try to infer company from filename
    # Covers patterns like: TCS_AnnualReport_2024, INFY_AR_2024,
    # annual-report-2023-2024 (fallback to filename stem)
    company_keywords = {
        "tcs": "TCS",
    "annual-report-2023-2024": "TCS",      # ← add this line
    "infosys": "Infosys", "infy": "Infosys",
    "wipro": "Wipro",
    "hcl": "HCL Technologies",
    "reliance": "Reliance",
    "hdfc": "HDFC Bank",
    "icici": "ICICI Bank",
    "axis": "Axis Bank",
    "bajfinance": "Bajaj Finance",          # ← add this line
    "bajaj": "Bajaj Finance",
    "asian": "Asian Paints",
    "lt_": "L&T", "larsen": "L&T",
    "maruti": "Maruti Suzuki",
    "sunpharma": "Sun Pharma", "sun_": "Sun Pharma",
    "itc": "ITC",
    "sbin": "SBI", "sbi": "SBI",           # ← add sbin
    "kotak": "Kotak",
    "airtel": "Bharti Airtel",
    "ntpc": "NTPC",
    "wipro": "Wipro",
    }

    for keyword, company in company_keywords.items():
        if keyword in name_lower:
            return "company", company

    # Fallback: use filename stem, truncated
    return "company", filepath.stem[:30]


def extract_pages(filepath: Path) -> Iterator[dict]:
    """
    Extract all pages from a PDF as cleaned, metadata-tagged dicts.

    Yields one dict per page. Pages that are:
      - Entirely empty after cleaning
      - Vertical design text (cover page decoration)
      - Too short to be meaningful (< 50 words)
    are silently skipped.
    """
    doc_type, entity = infer_doc_type(filepath)

    try:
        doc = fitz.open(filepath)
    except Exception as e:
        logger.error(f"Could not open {filepath.name}: {e}")
        return

    total_pages = len(doc)
    logger.info(f"Processing {filepath.name} ({total_pages} pages, {doc_type}, {entity})")

    for page_idx in range(total_pages):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        table_text = extract_tables_as_text(page)

        if table_text:
            raw_text = raw_text + "\n\n[TABLE DATA]\n" + table_text

        # Skip entirely empty pages (common in PDF dividers/section breaks)
        if not raw_text.strip():
            continue

        # Skip vertical design text pages (BSE cover pages)
        if is_vertical_design_text(raw_text):
            logger.debug(f"  Skipping page {page_idx+1}: vertical design text")
            continue

        clean_text = clean_page_text(raw_text)

        # Skip pages that are too short after cleaning
        word_count = len(clean_text.split())
        if word_count < 40:
            logger.debug(f"  Skipping page {page_idx+1}: only {word_count} words after cleaning")
            continue

        yield {
            "text":        clean_text,
            "source_file": filepath.name,
            "page_number": page_idx + 1,
            "doc_type":    doc_type,
            "company":     entity,
            "word_count":  word_count,
        }

    doc.close()


def load_all_documents(raw_dir: Path) -> list[dict]:
    """
    Load and clean every PDF in raw_dir and its subdirectories.

    Returns a flat list of page dicts across all documents.
    Prints a progress summary as it goes.
    """
    pdf_files = sorted(raw_dir.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    print(f"Loading {len(pdf_files)} PDFs from {raw_dir}")
    print("-" * 55)

    all_pages = []
    stats = {"files": 0, "pages_raw": 0, "pages_kept": 0, "skipped_files": 0}

    for pdf_path in pdf_files:
        pages_before = len(all_pages)

        try:
            for page_dict in extract_pages(pdf_path):
                all_pages.append(page_dict)

            pages_extracted = len(all_pages) - pages_before
            stats["files"] += 1
            stats["pages_kept"] += pages_extracted

            # Quick per-file feedback
            doc = fitz.open(pdf_path)
            raw_count = len(doc)
            doc.close()
            stats["pages_raw"] += raw_count

            print(f"  OK  {pdf_path.name[:55]:<55} "
                  f"{pages_extracted:>4} / {raw_count} pages kept")

        except Exception as e:
            logger.error(f"Failed: {pdf_path.name}: {e}")
            stats["skipped_files"] += 1
            print(f"  ERR {pdf_path.name[:55]:<55} {e}")

    print("-" * 55)
    print(f"Files processed : {stats['files']} / {len(pdf_files)}")
    print(f"Pages raw       : {stats['pages_raw']}")
    print(f"Pages kept      : {stats['pages_kept']} "
          f"({100*stats['pages_kept']//max(stats['pages_raw'],1)}% retention)")
    print(f"Avg words/page  : "
          f"{sum(p['word_count'] for p in all_pages) // max(len(all_pages), 1)}")

    return all_pages
