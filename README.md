# Finance RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for Q&A over Indian company annual reports and RBI publications.

## Demo

Ask questions in plain English. Get grounded answers with exact page citations.

**Example queries:**
- "What was RBI's repo rate decision in February 2025?"
- "What are Reliance Industries' main business segments?"
- "What was India's headline inflation in 2024-25?"
- "What is TCS's primary business?"
- "How many employees does Infosys have?"

## Results

| Metric | Score |
|--------|-------|
| Answer rate | 15/15 (100%) |
| Avg retrieval relevance | 0.822 |
| Hallucination rate | 0% |
| Total documents | 50 |
| Total pages processed | 9,155 |
| Total chunks | 33,229 |
| Vector store size | 33,689 |

## System Architecture

```
50 PDFs (RBI publications + Indian company annual reports)
        |
        v
PyMuPDF parser + custom table extractor
(strips headers, footers, page numbers, vertical design text)
        |
        v
RecursiveCharacterTextSplitter
(1024 tokens, 128 overlap, sentence-boundary aware)
        |
        v
BAAI/bge-small-en-v1.5 embeddings (384 dimensions)
        |
        v
ChromaDB persistent vector store (HNSW index, cosine similarity)
        |
        v
Hybrid retrieval: 70% dense vector + 30% BM25 keyword search
        |
        v
Llama-3.1-8b-instant via Groq API
(grounded system prompt, zero hallucination)
        |
        v
FastAPI REST backend + Streamlit chat UI
```

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Document parsing | PyMuPDF 1.24 | Handles multi-column layouts and table extraction |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects sentence boundaries, configurable overlap |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, local, within 5% of OpenAI on financial text |
| Vector store | ChromaDB 0.5 | Persistent, metadata filtering, no infra needed |
| Sparse retrieval | rank-bm25 | Catches financial acronyms pure vector search misses |
| LLM | Llama-3.1-8b-instant (Groq) | Free API, fast inference, strong instruction following |
| Evaluation | RAGAS | Faithfulness, relevance, precision, recall metrics |
| API | FastAPI 0.115 | Async, auto OpenAPI docs, Pydantic validation |
| UI | Streamlit 1.40 | Chat interface with source citations |

## Corpus

**Company annual reports (16 documents):**
TCS, Infosys, Wipro, HCL Technologies, Reliance Industries, ICICI Bank,
Axis Bank, Bajaj Finance, Asian Paints, L&T, Maruti Suzuki, Sun Pharma,
ITC, SBI, Kotak Mahindra Bank, Bharti Airtel

**RBI publications (34 documents):**
Annual Reports 2021-22 through 2024-25, Monetary Policy Reports,
Financial Stability Reports, Economic Review chapters,
Monetary Policy Operations chapters

**Total: 50 documents, 584MB, 9,155 pages**

## Setup

### Prerequisites
- Python 3.11+
- GROQ_API_KEY (free at console.groq.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/preetham789/finance-rag.git
cd finance-rag

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env
# Open .env and add your GROQ_API_KEY
```

### Data Collection

```bash
# Create directory structure
python scripts\setup_dirs.py

# Download RBI publications (automated)
python scripts\collect_rbi.py

# Open company IR pages in browser (semi-manual, ~20 mins)
python scripts\collect_bse.py --guide
python scripts\collect_bse.py --organize

# Verify corpus
python scripts\inventory.py
```

### Build the Vector Store

```bash
# Parse PDFs and create chunks
python scripts\ingest.py --strategy recursive

# Compare chunking strategies (optional)
python scripts\ingest.py --compare

# Embed chunks into ChromaDB
python scripts\embed.py
```

### Run the System

```bash
# Start API and UI together
python scripts\start.py

# Or separately
python scripts\start.py --api    # API at http://localhost:8000
python scripts\start.py --ui     # UI  at http://localhost:8501
```

- **Chat UI:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Ask a question (no filter)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was RBI repo rate in February 2025?"}'

# Ask with company filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main business segments?", "company": "Reliance"}'

# Ask with document type filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was CRR in December 2024?", "doc_type": "rbi"}'
```

**Response format:**
```json
{
  "question": "What was RBI repo rate in February 2025?",
  "answer": "The RBI reduced the policy repo rate by 25 basis points to 6.25 per cent in February 2025 [Source: RBI, Page 112].",
  "sources": [
    {
      "company": "RBI",
      "page_number": 112,
      "source_file": "0ANNUALREPORT202425.pdf",
      "score": 0.810,
      "preview": "The monetary policy committee reduced the policy repo rate..."
    }
  ],
  "model": "llama-3.1-8b-instant",
  "tokens_used": 1382,
  "latency_ms": 1026
}
```

## Evaluation

```bash
# Run evaluation (answers only, no API cost)
python scripts\evaluate.py --answers-only --save

# Full RAGAS evaluation (requires OpenAI API key, ~$0.02)
python scripts\evaluate.py --save
```

**Baseline results (eval_20260426_1625.json):**

| Question | Score | Status |
|----------|-------|--------|
| RBI repo rate Feb 2025 | 0.810 | Answered |
| MPC stance Oct 2024 | 0.786 | Answered |
| CRR decision Dec 2024 | 0.809 | Answered |
| Global GDP growth 2024 | 0.848 | Answered |
| Global inflation 2024 | 0.846 | Answered |
| India headline inflation 2024-25 | 0.869 | Answered |
| Repo rate Feb 2023 to 2025 | 0.813 | Answered |
| TCS primary business | 0.824 | Answered |
| TCS employee count | 0.811 | Answered |
| Reliance business segments | 0.794 | Answered |
| Infosys business model | 0.817 | Answered |
| Maruti Suzuki core business | 0.819 | Answered |
| HCL Technologies focus areas | 0.815 | Answered |
| ITC business portfolio | 0.812 | Answered |
| Wipro service offering | 0.803 | Answered |

**Answer rate: 15/15 (100%) | Average relevance: 0.822**

## Project Structure

```
finance-rag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF parser with table extractor
│   │   ├── chunker.py         # Three chunking strategies
│   │   └── embedder.py        # BGE embeddings + ChromaDB
│   ├── retrieval/
│   │   └── retriever.py       # Hybrid dense + BM25 search
│   ├── generation/
│   │   └── chain.py           # RAG chain, multi-provider LLM
│   ├── evaluation/
│   │   └── ragas_eval.py      # 15-question test dataset + RAGAS
│   ├── api/
│   │   ├── main.py            # FastAPI backend
│   │   └── streamlit_app.py   # Streamlit chat UI
│   └── config.py              # Central configuration
├── scripts/
│   ├── collect_rbi.py         # RBI document downloader
│   ├── collect_bse.py         # Company IR page guide
│   ├── ingest.py              # Full ingestion pipeline
│   ├── embed.py               # Embedding pipeline
│   ├── query.py               # CLI query interface
│   ├── evaluate.py            # Evaluation runner
│   ├── inventory.py           # Corpus statistics
│   └── start.py               # System launcher
├── data/
│   ├── raw/                   # Source PDFs (gitignored)
│   │   ├── rbi/               # RBI publications
│   │   └── bse/               # Company annual reports
│   ├── processed/             # Chunks JSONL (gitignored)
│   └── test_sets/             # Evaluation results (JSON)
├── experiments/               # ChromaDB vectors (gitignored)
├── .env.example               # API key template
├── requirements.txt
└── README.md
```

## Key Engineering Decisions

**Hybrid retrieval (dense + BM25)**
Financial queries contain specific acronyms — NPA, GNPA, CRR, NDTL, repo rate —
that pure vector search frequently misses because embeddings generalize meaning.
BM25 keyword scoring handles exact-match terms. Alpha=0.7 (70% dense, 30% BM25)
was chosen empirically as the best balance for financial Q&A.

**Table-aware PDF parsing**
Standard `get_text()` extraction loses column headers from financial tables.
A table like `Gross NPA (%) | 1.24 | 1.34` becomes `1.24 1.34` with no labels.
Custom block-level extractor reconstructs `Gross NPA (%): 1.24` key-value pairs,
making financial ratios retrievable.

**Corpus validation catches real bugs**
Mid-project discovery: HDFC Securities (brokerage, 197 pages) was downloaded
instead of HDFC Bank (lender, ~500 pages). The evaluation framework made this
detectable — all HDFC NPA queries returned "insufficient information" despite
high relevance scores. Lesson: always validate corpus content, not just file count.

**Grounded system prompt**
The LLM is instructed to answer ONLY from retrieved context and say
"The provided documents do not contain sufficient information" when context
is absent. Verified with out-of-corpus query ("What is Apple's revenue?") —
system correctly refused to answer.

**Provider-agnostic LLM client**
The chain supports Groq (free), OpenAI, and Ollama (local) via the same
OpenAI-compatible API interface. Switching providers requires changing one
environment variable, not rewriting code.

## What I Learned

The hardest part of RAG is not the LLM — it is data quality.
Chunking strategy, metadata tagging, table extraction, and corpus
validation account for approximately 70% of system quality.

Key lessons from building this:
- Corpus validation is as important as model selection
- Hybrid retrieval consistently outperforms pure vector search on domain-specific text
- RAGAS evaluation turns subjective quality judgement into measurable, comparable metrics
- Table data in PDFs requires separate extraction logic from prose text
- Provider-agnostic design makes the system portable and cost-flexible


