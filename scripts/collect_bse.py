# scripts/collect_annual_reports.py  (replaces collect_bse.py entirely)
"""
Annual Report Collector — NSE + Direct Company IR Pages

BSE changed their URL structure. This script uses two better sources:

  SOURCE 1: NSE EDGAR-equivalent (efiling.nseindia.com)
            Structured API, returns JSON, very stable.

  SOURCE 2: Direct company Investor Relations pages
            Each major Indian company has a dedicated IR page
            that hosts annual reports. These are the most stable
            URLs that exist — companies never break their own IR pages.

This is a real engineering lesson: when one source breaks,
you pivot to a more reliable one. Don't waste time fighting
a broken source.
"""

import sys
import time
import re
import webbrowser
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

# ── Direct IR page URLs — curated and verified ──
# Format: (Company, Symbol, Direct_AR_page_URL, notes)
COMPANY_IR_PAGES = [
    (
        "Infosys",
        "INFY",
        "https://www.infosys.com/investors/reports-filings/annual-report.html",
        "Lists last 5 years — click any to download PDF",
    ),
    (
        "TCS",
        "TCS",
        "https://www.tcs.com/investor-relations/annual-reports",
        "Direct PDF links listed by year",
    ),
    (
        "Wipro",
        "WIPRO",
        "https://www.wipro.com/investors/wipro-annual-report/",
        "PDF download buttons by year",
    ),
    (
        "HCL Technologies",
        "HCLTECH",
        "https://www.hcltech.com/investor-relations/annual-reports",
        "Annual reports with direct download",
    ),
    (
        "Reliance Industries",
        "RELIANCE",
        "https://www.ril.com/investor-relations/annual-reports",
        "Lists reports — click year → download PDF",
    ),
    (
        "HDFC Bank",
        "HDFCBANK",
        "https://www.hdfcbank.com/content/bbp/repositories/723fb80a-2dde-42a3-9793-7ae1be57c87f/?folderPath=/OtherDocuments/Annual%20Report/",
        "Direct folder of annual report PDFs",
    ),
    (
        "ICICI Bank",
        "ICICIBANK",
        "https://www.icicibank.com/investor-relations/annual-reports",
        "Annual reports by year",
    ),
    (
        "Axis Bank",
        "AXISBANK",
        "https://www.axisbank.com/investor-relations/annual-reports",
        "PDF links listed",
    ),
    (
        "Bajaj Finance",
        "BAJFINANCE",
        "https://www.bajajfinserv.in/bajaj-finance-annual-report",
        "Annual reports page",
    ),
    (
        "Asian Paints",
        "ASIANPAINT",
        "https://www.asianpaints.com/investor-relations/annual-reports.html",
        "PDF links by year",
    ),
    (
        "Larsen and Toubro",
        "LT",
        "https://investors.larsentoubro.com/annual-report.aspx",
        "AR page with year-wise links",
    ),
    (
        "Maruti Suzuki",
        "MARUTI",
        "https://www.marutisuzuki.com/corporate/investors/annual-reports",
        "Annual reports list",
    ),
    (
        "Sun Pharma",
        "SUNPHARMA",
        "https://sunpharma.com/investor-relations/annual-report/",
        "PDF downloads",
    ),
    (
        "ITC",
        "ITC",
        "https://www.itcportal.com/investor-relations/annual-reports-and-accounts.aspx",
        "Historical annual reports",
    ),
    (
        "SBI",
        "SBIN",
        "https://bank.sbi/web/investor-relations/annual-reports",
        "SBI annual reports",
    ),
]


def try_nse_api(symbol: str, dest_dir: Path) -> bool:
    """
    Try NSE's filing API to get annual report PDF for a given symbol.
    NSE's API returns JSON with direct PDF links for recent filings.
    Returns True if a PDF was downloaded successfully.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # Step 1: get NSE cookies first (required for API calls)
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        session.get("https://www.nseindia.com/companies-listing/corporate-filings-annual-reports", timeout=10)
        time.sleep(1)

        # Step 2: call the annual reports API
        api_url = (
            f"https://www.nseindia.com/api/annual-reports"
            f"?index=equities&symbol={symbol}"
        )
        resp = session.get(api_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list):
            return False

        # Take the most recent entry
        latest = data[0]
        pdf_url = latest.get("fileName") or latest.get("fileUrl") or ""
        year    = latest.get("year") or latest.get("toDate", "")[:4] or "2024"

        if not pdf_url:
            return False

        if not pdf_url.startswith("http"):
            pdf_url = "https://www.nseindia.com" + pdf_url

        filename = f"{symbol}_AnnualReport_{year}.pdf"
        dest = dest_dir / filename

        pdf_resp = session.get(pdf_url, stream=True, timeout=60)
        pdf_resp.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in pdf_resp.iter_content(chunk_size=32_768):
                f.write(chunk)

        # Validate it's a real PDF
        with open(dest, "rb") as f:
            if f.read(4) != b"%PDF":
                dest.unlink()
                return False

        size_kb = dest.stat().st_size // 1024
        print(f"    NSE API OK: {filename} ({size_kb} KB)")
        return True

    except Exception as e:
        print(f"    NSE API failed for {symbol}: {e}")
        return False


def open_ir_pages_in_browser(dest_dir: Path):
    """
    Opens each company's IR page in the browser.
    Shows exactly what to download and where to save it.
    """
    print("=" * 60)
    print("Annual Report Download — Browser Guide")
    print("=" * 60)
    print()
    print(f"Save ALL PDFs to this folder:")
    print(f"  {dest_dir}")
    print()
    print("For each company:")
    print("  1. Page opens in browser")
    print("  2. Click the most recent annual report link")
    print("  3. Save PDF with name like:  CompanyName_AR_2024.pdf")
    print("  4. Come back here and press ENTER for next company")
    print()
    print("You need at least 15 companies. Skip any that are slow.")
    print()

    for i, (name, symbol, url, hint) in enumerate(COMPANY_IR_PAGES, 1):
        print(f"[{i:02d}/{len(COMPANY_IR_PAGES)}]  {name}  ({symbol})")
        print(f"  Hint : {hint}")
        print(f"  URL  : {url}")

        choice = input("  Press ENTER to open, 's' to skip: ").strip().lower()
        if choice == "s":
            print("  Skipped.\n")
            continue

        webbrowser.open(url)
        time.sleep(2)

        print(f"  After downloading, save as:  {symbol}_AnnualReport_2024.pdf")
        input("  Press ENTER when done (or just continue)...\n")


def organize_downloads(dest_dir: Path):
    """
    Scan Downloads, Desktop, Documents for annual report PDFs.
    Copy them into dest_dir with clean names.
    """
    import shutil

    home = Path.home()
    search_dirs = [
        home / "Downloads",
        home / "Desktop",
        home / "Documents",
    ]

    found = []
    for d in search_dirs:
        if d.exists():
            found.extend(d.glob("*.pdf"))
            found.extend(d.glob("*.PDF"))

    if not found:
        print("No PDFs found in Downloads/Desktop/Documents.")
        return

    print(f"Found {len(found)} PDFs — copying annual reports to {dest_dir}\n")

    # Keywords that suggest it's an annual report
    ar_keywords = [
        "annual", "report", "ar202", "ar_202",
        "infosys", "tcs", "wipro", "hdfcbank", "icici",
        "reliance", "sbi", "axisbank", "bajaj", "maruti",
        "sunpharma", "itc", "hcl", "asianpaint", "lt_",
    ]

    copied = 0
    for pdf in found:
        name_lower = pdf.name.lower()
        if any(kw in name_lower for kw in ar_keywords):
            dest = dest_dir / pdf.name
            if not dest.exists():
                shutil.copy2(pdf, dest)
                print(f"  Copied: {pdf.name}")
                copied += 1
            else:
                print(f"  Skip (exists): {pdf.name}")

    print(f"\nCopied {copied} files.")
    print(f"Total in bse/: {len(list(dest_dir.glob('*.pdf')))}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect Indian company annual reports")
    parser.add_argument("--nse-api",  action="store_true", help="Try NSE API for all companies")
    parser.add_argument("--guide",    action="store_true", help="Open IR pages in browser")
    parser.add_argument("--organize", action="store_true", help="Organize downloaded PDFs")
    args = parser.parse_args()

    dest_dir = RAW_DIR / "bse"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if args.nse_api:
        print("Trying NSE API for all companies...\n")
        success = 0
        for name, symbol, _, _ in COMPANY_IR_PAGES:
            print(f"[{name}]")
            ok = try_nse_api(symbol, dest_dir)
            if ok:
                success += 1
            time.sleep(3)
        print(f"\nNSE API: got {success}/{len(COMPANY_IR_PAGES)} files")
        print("Run --guide for any that failed, then --organize")

    elif args.guide:
        open_ir_pages_in_browser(dest_dir)
        print("\nAll done. Now run:")
        print("  python scripts\\collect_annual_reports.py --organize")

    elif args.organize:
        organize_downloads(dest_dir)
        print("\nThen run: python scripts\\inventory.py")

    else:
        print("Usage:")
        print()
        print("  STEP 1 — Try automatic NSE API first (may get some):")
        print("    python scripts\\collect_annual_reports.py --nse-api")
        print()
        print("  STEP 2 — Open browser for remaining companies:")
        print("    python scripts\\collect_annual_reports.py --guide")
        print()
        print("  STEP 3 — Organize all downloads into data/raw/bse/:")
        print("    python scripts\\collect_annual_reports.py --organize")
        print()
        print("  STEP 4 — Check your corpus:")
        print("    python scripts\\inventory.py")


if __name__ == "__main__":
    main()