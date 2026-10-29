from src.ingestion.chunker import (
    make_sentence_chunks,
    page_dicts_to_chunks,
    split_into_sentences,
)


def test_split_into_sentences_preserves_abbreviations():
    text = (
        "The repo rate was kept at 6.50% by the MPC, viz. unchanged. "
        "Gross NPAs of banks declined to 2.8% as per RBI. "
        "Revenue was Rs. 15,000 Cr. for the full fiscal year."
    )
    sents = split_into_sentences(text)
    assert len(sents) == 3
    assert "viz." in sents[0]
    assert "Rs." in sents[2]
    assert "Cr." in sents[2]


def test_make_sentence_chunks_overlap():
    text = (
        "Sentence one is here. "
        "Sentence two follows it. "
        "Sentence three provides data. "
        "Sentence four concludes paragraph."
    )
    chunks = make_sentence_chunks(text, max_sentences=2, overlap_sentences=1)
    assert len(chunks) >= 2
    # Ensure overlap sentence is present in adjacent chunks
    assert "Sentence two" in chunks[0]
    assert "Sentence two" in chunks[1]


def test_page_dicts_to_chunks_with_contextual_headers():
    pages = [
        {
            "text": "Gross NPA ratio was 1.24% as of March 31, 2024. Net NPA was 0.33%. Capital adequacy remained strong at 18.8%.",
            "source_file": "HDFCBANK_AnnualReport_2024.pdf",
            "page_number": 14,
            "doc_type": "company",
            "company": "HDFC Bank",
            "fiscal_year": "2024",
            "report_type": "annual_report",
        }
    ]

    chunks = page_dicts_to_chunks(pages, strategy="recursive", add_contextual_header=True)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c["company"] == "HDFC Bank"
    assert c["fiscal_year"] == "2024"
    assert c["report_type"] == "annual_report"
    assert c["chunk_type"] == "prose"
    # Check that contextual header is prepended
    assert c["text"].startswith("[HDFC Bank | ANNUAL_REPORT | 2024 | Page 14]")
