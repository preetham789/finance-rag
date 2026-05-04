# scripts/collect_rbi.py
"""
RBI collector — targeted at the exact page structure visible on rbi.org.in.

Strategy:
  1. Download full annual report PDFs from archive years (2020-2024)
  2. Download high-value individual chapter PDFs from 2024-25
  3. Grab Monetary Policy Reports and FSR from their listing pages

This script targets the real HTML structure of RBI's publication pages,
not guessed URLs. It is resilient to URL changes because it reads links
directly from the page — the same way your browser does.
"""

import sys
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR

# ── Browser session headers ──
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.rbi.org.in/",
}

# ── Publication listing pages ──
# These are the stable index URLs — we scrape PDF links FROM these.
LISTING_PAGES = [
    {
        "name": "Annual Report",
        "url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?head=Annual+Report",
        "prefix": "RBI_AR",
        "max_files": 6,       # grab last 6 years
        "min_size_kb": 500,   # skip tiny chapter fragments — full report only
    },
    {
        "name": "Monetary Policy Report",
        "url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?head=Monetary%20Policy%20Report",
        "prefix": "RBI_MPR",
        "max_files": 5,
        "min_size_kb": 100,
    },
    {
        "name": "Financial Stability Report",
        "url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?head=Financial%20Stability%20Report",
        "prefix": "RBI_FSR",
        "max_files": 4,
        "min_size_kb": 100,
    },
    {
        "name": "Report on Currency and Finance",
        "url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?head=Report+on+Currency+and+Finance",
        "prefix": "RBI_RCF",
        "max_files": 3,
        "min_size_kb": 100,
    },
]

# ── Chapter PDFs from the 2024-25 annual report ──
# These are the specific chapters visible in your screenshot.
# We include only the content-rich chapters — skip front matter.
# Based on the structure seen on: rbi.org.in → Annual Report 2024-25
CHAPTER_PAGES = [
    {
        "name": "RBI Annual Report 2024-25 chapters",
        "base_url": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx?head=Annual+Report",
        "chapters_to_grab": [
            "Assessment and Prospects",
            "Economic Review",
            "Monetary Policy Operations",
            "Credit Delivery and Financial Inclusion",
            "Financial Markets",
            "Regulation, Supervision and Financial Stability",
            "Reserve Bank",
        ],
    }
]


def make_session() -> requests.Session:
    """Single session maintains cookies across all requests — like a real browser."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_page(session: requests.Session, url: str) -> BeautifulSoup | None:
    """Fetch a page and return parsed HTML. Returns None on failure."""
    try:
        resp = session.get(url, timeout=25)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.exceptions.Timeout:
        print(f"    TIMEOUT loading: {url[:60]}")
    except requests.exceptions.HTTPError as e:
        print(f"    HTTP {e.response.status_code}: {url[:60]}")
    except Exception as e:
        print(f"    ERROR: {e}")
    return None


def extract_pdf_links(soup: BeautifulSoup, base_url: str, min_size_kb: int = 0) -> list[tuple[str, str, int]]:
    """
    Extract all PDF links from a RBI publication page.

    RBI pages follow a pattern: each row has a chapter name in a <td>,
    a PDF icon image, and an <a href> pointing to the PDF. Some rows
    also show file size in KB.

    Returns list of (chapter_name, absolute_pdf_url, size_kb)
    """
    results = []
    base = "https://www.rbi.org.in"

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        # Only care about PDF links
        if not href.lower().endswith(".pdf"):
            continue

        # Build absolute URL
        if href.startswith("http"):
            pdf_url = href
        elif href.startswith("//"):
            pdf_url = "https:" + href
        else:
            pdf_url = base + ("/" if not href.startswith("/") else "") + href

        # Try to get the chapter name from surrounding text
        parent = a_tag.find_parent("tr")
        if parent:
            # Get all text in this table row, strip whitespace
            row_text = parent.get_text(" ", strip=True)
            # Remove KB size info and PDF label noise
            name = re.sub(r"\d+\s*kb", "", row_text, flags=re.IGNORECASE).strip()
            name = re.sub(r"\s{2,}", " ", name).strip()
        else:
            name = a_tag.get_text(strip=True) or Path(href).stem

        # Try to extract size from row text
        size_kb = 0
        if parent:
            size_match = re.search(r"(\d+)\s*kb", parent.get_text(), re.IGNORECASE)
            if size_match:
                size_kb = int(size_match.group(1))

        if min_size_kb > 0 and 0 < size_kb < min_size_kb:
            continue  # skip small files (front matter, contents page, etc.)

        results.append((name[:80], pdf_url, size_kb))

    return results


def sanitize_filename(text: str) -> str:
    """Convert arbitrary text to a safe Windows filename."""
    safe = re.sub(r'[\\/*?:"<>|]', "_", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:80]


def download_pdf(session: requests.Session, url: str, dest: Path) -> bool:
    """
    Download a PDF file. Uses streaming to handle large files safely.
    Validates the result is actually a PDF (not an HTML error page).
    """
    if dest.exists() and dest.stat().st_size > 50_000:
        print(f"    skip: {dest.name}  (already exists)")
        return True

    try:
        # Send Referer header so RBI server accepts the request
        headers = {"Referer": "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx"}
        resp = session.get(url, stream=True, timeout=60, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type and "pdf" not in content_type:
            print(f"    WARN: server sent HTML instead of PDF for {dest.name}")
            return False

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=32_768):
                f.write(chunk)

        size_kb = dest.stat().st_size // 1024

        # A real PDF starts with the bytes %PDF — check for this
        with open(dest, "rb") as f:
            magic = f.read(4)
        if magic != b"%PDF":
            dest.unlink()
            print(f"    INVALID: {dest.name} is not a real PDF (got {magic})")
            return False

        print(f"    OK  {dest.name}  ({size_kb} KB)")
        return True

    except requests.exceptions.Timeout:
        print(f"    TIMEOUT: {dest.name}")
    except Exception as e:
        print(f"    ERROR {dest.name}: {e}")
    return False


def collect_listing_page(session, pub, rbi_dir) -> tuple[int, list]:
    """Scrape one publication listing page and download PDFs found there."""
    print(f"\n[{pub['name']}]")
    print(f"  Scanning: {pub['url']}")

    soup = get_page(session, pub["url"])
    if not soup:
        print("  Could not load page — skipping")
        return 0, []

    links = extract_pdf_links(soup, pub["url"], min_size_kb=pub.get("min_size_kb", 0))
    print(f"  Found {len(links)} PDF link(s)")

    if not links:
        print("  No PDFs found on this listing page")
        return 0, []

    success = 0
    failed = []

    for i, (name, url, size_kb) in enumerate(links[:pub["max_files"]]):
        filename = f"{pub['prefix']}_{sanitize_filename(name)}.pdf"
        dest = rbi_dir / filename
        size_str = f"{size_kb} KB" if size_kb else "size unknown"
        print(f"  [{i+1}] {name[:50]}  ({size_str})")

        ok = download_pdf(session, url, dest)
        if ok:
            success += 1
        else:
            failed.append(name)

        time.sleep(2)   # 2 second pause between downloads

    return success, failed


def main():
    rbi_dir = RAW_DIR / "rbi"
    rbi_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("RBI Publication Collector")
    print(f"Destination: {rbi_dir}")
    print("=" * 55)

    session = make_session()

    # Warm up session by visiting RBI homepage first
    print("\nWarming up session...")
    try:
        session.get("https://www.rbi.org.in", timeout=15)
        print("  Done.\n")
    except Exception:
        print("  Could not reach homepage — trying anyway.\n")
    time.sleep(2)

    total_success = 0
    total_failed = []

    for pub in LISTING_PAGES:
        s, f = collect_listing_page(session, pub, rbi_dir)
        total_success += s
        total_failed.extend(f)
        time.sleep(3)   # pause between publication types

    # ── Final summary ──
    print(f"\n{'='*55}")
    all_pdfs = list(rbi_dir.glob("*.pdf"))
    total_mb = sum(f.stat().st_size for f in all_pdfs) / (1024 * 1024)

    print(f"Downloaded this run : {total_success} files")
    print(f"Total in rbi/       : {len(all_pdfs)} files")
    print(f"Total size          : {total_mb:.1f} MB")

    if total_failed:
        print(f"\nFailed ({len(total_failed)}): {', '.join(total_failed[:5])}")

    if len(all_pdfs) < 8:
        print("\n--- MANUAL DOWNLOAD GUIDE ---")
        print("The automated scraper got fewer than 8 files.")
        print("Do this — takes 5 minutes:")
        print()
        print("You are already on the right page (your screenshot).")
        print("From that page, RIGHT-CLICK each PDF icon → Save link as")
        print("Save directly into:")
        print(f"  {rbi_dir}")
        print()
        print("Priority chapters to download from Annual Report 2024-25:")
        priority = [
            "Annual Report 2024-25 (full, 7874 KB)",
            "II. Economic Review (4570 KB)",
            "III. Monetary Policy Operations (723 KB)",
            "VI. Regulation, Supervision and Financial Stability (352 KB)",
            "XII. The Reserve Bank's Accounts for 2024-25 (422 KB)",
        ]
        for p in priority:
            print(f"    - {p}")
        print()
        print("Then do the same for 2023 and 2022 from the right panel.")
        print("That gives you 15+ high-quality RBI documents instantly.")
    else:
        print("\nRBI collection complete.")
        print("Run: python scripts\\inventory.py")


if __name__ == "__main__":
    main()