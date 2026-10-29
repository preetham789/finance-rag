from pathlib import Path
from src.ingestion.loader import (
    clean_page_text,
    compute_file_hash,
    infer_metadata,
    is_noise_line,
    is_vertical_design_text,
)


def test_is_noise_line():
    assert is_noise_line("17")
    assert is_noise_line(" 42 ")
    assert is_noise_line("ANNUAL REPORT 2023-24")
    assert is_noise_line("MONETARY POLICY REPORT")
    assert is_noise_line("ECONOMIC REVIEW")
    assert is_noise_line("BSE Limited")
    assert is_noise_line("---")
    assert is_noise_line("===")
    assert is_noise_line("See page 42")
    # Real financial prose should not be classified as noise
    assert not is_noise_line("Net Interest Margin expanded to 4.3% in FY24.")
    assert not is_noise_line("The Monetary Policy Committee decided to keep repo rate unchanged.")


def test_is_vertical_design_text():
    # Normal text
    assert not is_vertical_design_text("This is standard horizontal annual report text.")

    # Vertical stacked letters
    vertical = "T R A N S F O R M I N G I N D U S T R I E S"
    assert is_vertical_design_text(vertical)


def test_clean_page_text():
    raw = """12
    
    ANNUAL REPORT 2024
    
    TCS reported consolidated revenue of Rs 240,893 crore.
    ===
    Growth was broad-based across all verticals.
    """
    cleaned = clean_page_text(raw)
    assert "12" not in cleaned
    assert "ANNUAL REPORT 2024" not in cleaned
    assert "===" not in cleaned
    assert "TCS reported consolidated revenue" in cleaned
    assert "Growth was broad-based" in cleaned


def test_infer_metadata():
    doc_type, entity, fy, report_type = infer_metadata(Path("data/raw/bse/HDFCBANK_AnnualReport_2024.pdf"))
    assert doc_type == "company"
    assert entity == "HDFC Bank"
    assert "2024" in fy
    assert report_type == "annual_report"

    doc_type, entity, fy, report_type = infer_metadata(Path("data/raw/rbi/03MONETARY29052025.pdf"))
    assert doc_type == "rbi"
    assert entity == "RBI"
    assert "2025" in fy
    assert report_type == "monetary_policy"


def test_compute_file_hash(tmp_path):
    f1 = tmp_path / "doc1.pdf"
    f2 = tmp_path / "doc2.pdf"
    f3 = tmp_path / "doc3.pdf"

    f1.write_bytes(b"identical content")
    f2.write_bytes(b"identical content")
    f3.write_bytes(b"different content")

    h1 = compute_file_hash(f1)
    h2 = compute_file_hash(f2)
    h3 = compute_file_hash(f3)

    assert h1 == h2
    assert h1 != h3
