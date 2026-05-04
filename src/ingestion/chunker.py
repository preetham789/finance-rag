# src/ingestion/chunker.py
"""
Chunker — splits cleaned page text into chunks ready for embedding.

Implements 3 strategies so you can compare them with RAGAS in Phase 5:
  1. recursive  — LangChain RecursiveCharacterTextSplitter  (your baseline)
  2. token      — TokenTextSplitter (token-aware, no mid-token cuts)
  3. sentence   — sentence-boundary aware, good for financial Q&A

Each strategy takes a list of page dicts (from loader.py) and returns
a list of chunk dicts — same metadata fields, plus chunk-specific fields.

Why keep metadata on every chunk?
  When a user asks "What did HDFC Bank say about NPAs?", you want to
  retrieve chunks tagged company="HDFC Bank" — not all 53 documents.
  Metadata enables filtered retrieval, which is a key interview topic.
"""

import re
import logging
from typing import Literal

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

logger = logging.getLogger(__name__)

# ── Type alias for clarity ──
ChunkStrategy = Literal["recursive", "token", "sentence"]


def make_recursive_splitter(chunk_size: int = 512, chunk_overlap: int = 64):
    """
    RecursiveCharacterTextSplitter — the industry default starting point.

    Tries separators in order: paragraph → newline → sentence → word → char.
    For financial documents, we add ". " as an explicit separator because
    RBI reports use long paragraphs with dense information per sentence.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",    # paragraph break — strongest boundary
            "\n",      # line break
            ". ",      # sentence boundary — important for financial prose
            ", ",      # clause boundary — fallback
            " ",       # word boundary — last resort
            "",        # character — never reached in practice
        ],
        length_function=len,
        is_separator_regex=False,
    )


def make_token_splitter(chunk_size: int = 400, chunk_overlap: int = 50):
    """
    TokenTextSplitter — splits on actual token count, not character count.

    Why this matters:
      "GDP grew by 3.3 per cent" is 6 words but ~8 tokens (numbers tokenize
      differently). With character splitting, you might cut mid-token when
      the chunk hits exactly 512 chars. Token splitting prevents this.

    chunk_size=400 tokens ≈ 512 characters for English financial text.

    Requires: pip install tiktoken (already in requirements.txt)
    """
    try:
        from langchain.text_splitter import TokenTextSplitter
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",  # GPT-4 / text-embedding-3 tokenizer
        )
    except ImportError:
        logger.warning("tiktoken not installed — falling back to recursive splitter")
        return make_recursive_splitter()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation rules.
    Handles abbreviations common in financial text:
      "e.g.", "i.e.", "Rs.", "per cent.", "Fig.", "No.", "vs."
    """
    # Protect common abbreviations from being split
    protected = text
    abbrevs = ["e.g.", "i.e.", "viz.", "Rs.", "Cr.", "Fig.", "No.", "vs.",
                "per cent.", "pp.", "p.a.", "Ltd.", "Inc.", "Corp.", "Co."]
    placeholders = {}
    for i, abbrev in enumerate(abbrevs):
        placeholder = f"__ABBREV{i}__"
        placeholders[placeholder] = abbrev
        protected = protected.replace(abbrev, placeholder)

    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    # Restore abbreviations
    restored = []
    for sent in sentences:
        for placeholder, abbrev in placeholders.items():
            sent = sent.replace(placeholder, abbrev)
        sent = sent.strip()
        if sent:
            restored.append(sent)

    return restored


def make_sentence_chunks(text: str, max_sentences: int = 6,
                          overlap_sentences: int = 1) -> list[str]:
    """
    Group sentences into chunks of max_sentences, with overlap_sentences
    from the previous chunk carried forward.

    Why sentence-based chunking for finance:
      Financial statements and RBI reports express one complete idea per
      sentence. Cutting mid-sentence loses the subject or the predicate.
      This strategy guarantees every chunk starts and ends at a sentence
      boundary — making it the cleanest for factual Q&A retrieval.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i: i + max_sentences]
        chunk_text = " ".join(chunk_sentences)

        if len(chunk_text.split()) >= 20:  # skip tiny fragments
            chunks.append(chunk_text)

        # Advance by (max_sentences - overlap_sentences) to create overlap
        i += max(1, max_sentences - overlap_sentences)

    return chunks


def page_dicts_to_chunks(
    pages: list[dict],
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Convert a list of page dicts into a list of chunk dicts.

    Each chunk dict contains:
      text          — the chunk text
      chunk_id      — unique identifier  e.g. "HDFC_AR_2024_p12_c3"
      strategy      — which chunking strategy produced this chunk
      chunk_index   — position of this chunk within its source page
      + all metadata fields from the page dict (source_file, company, etc.)

    This is the function you'll call from scripts/ingest.py.
    """
    if strategy == "recursive":
        splitter = make_recursive_splitter(chunk_size, chunk_overlap)
    elif strategy == "token":
        splitter = make_token_splitter(
            chunk_size=chunk_size // 2,   # token count ≈ half char count
            chunk_overlap=chunk_overlap // 2,
        )
    # sentence strategy uses our custom function below

    all_chunks = []

    for page in pages:
        text = page["text"]
        if not text.strip():
            continue

        # ── Produce raw text chunks based on strategy ──
        if strategy in ("recursive", "token"):
            raw_chunks = splitter.split_text(text)
        elif strategy == "sentence":
            raw_chunks = make_sentence_chunks(text, max_sentences=6, overlap_sentences=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'recursive', 'token', or 'sentence'")

        # ── Tag every chunk with metadata ──
        for chunk_idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text.split()) < 15:
                continue  # drop micro-chunks (< 15 words — not useful for RAG)

            # Build a readable unique ID
            safe_company = re.sub(r"[^a-zA-Z0-9]", "_", page["company"])[:15]
            chunk_id = (
                f"{safe_company}"
                f"_p{page['page_number']}"
                f"_c{chunk_idx}"
                f"_{strategy[:3]}"
            )

            all_chunks.append({
                # ── Text ──
                "text":         chunk_text,
                "word_count":   len(chunk_text.split()),

                # ── Identity ──
                "chunk_id":     chunk_id,
                "strategy":     strategy,
                "chunk_index":  chunk_idx,

                # ── Source metadata (inherited from page) ──
                "source_file":  page["source_file"],
                "page_number":  page["page_number"],
                "doc_type":     page["doc_type"],
                "company":      page["company"],
            })

    return all_chunks


def compare_strategies(pages: list[dict], sample_pages: int = 10) -> None:
    """
    Run all 3 strategies on a sample of pages and print comparison stats.
    Call this to decide which strategy to use before full ingestion.
    """
    sample = pages[:sample_pages]
    print(f"\nChunking strategy comparison (on {len(sample)} sample pages)")
    print("=" * 60)

    for strategy in ("recursive", "token", "sentence"):
        chunks = page_dicts_to_chunks(sample, strategy=strategy)
        if not chunks:
            print(f"  {strategy:<12}: 0 chunks produced")
            continue

        word_counts = [c["word_count"] for c in chunks]
        avg_words   = sum(word_counts) // len(word_counts)
        min_words   = min(word_counts)
        max_words   = max(word_counts)

        print(f"  {strategy:<12}: {len(chunks):>4} chunks | "
              f"avg {avg_words:>4} words | "
              f"min {min_words:>3} | max {max_words:>4}")

    print("=" * 60)
    print("Recommendation: start with 'recursive', switch to 'sentence'")
    print("if RAGAS context_precision < 0.6 in Phase 5.\n")