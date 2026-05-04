# scripts/peek_pdf.py  — run this once, then we can delete it
"""Quick diagnostic — shows what text PyMuPDF extracts from your first 2 PDFs."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DIR
import fitz  # PyMuPDF

for subfolder in ["rbi", "bse"]:
    folder = RAW_DIR / subfolder
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        continue
    pdf_path = pdfs[0]
    print(f"\n{'='*55}")
    print(f"File: {pdf_path.name}")
    print(f"{'='*55}")
    doc = fitz.open(pdf_path)
    print(f"Pages: {len(doc)}")
    for page_num in range(min(2, len(doc))):
        page = doc[page_num]
        text = page.get_text("text")
        print(f"\n--- Page {page_num+1} (first 500 chars) ---")
        print(repr(text[:500]))
    doc.close()