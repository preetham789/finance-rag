# Building a Finance RAG System for Indian Annual Reports and RBI Publications

## Introduction

Financial documents are full of useful answers, but they are not always easy to search.

Annual reports, RBI publications, monetary policy reports, and financial stability reports can run into hundreds of pages. A simple keyword search may miss the meaning of a question, while a normal chatbot may answer confidently without knowing the source.

That is why I built a Finance RAG system.

This project answers natural language questions over Indian company annual reports and RBI publications. The goal is simple: ask a finance question, get a grounded answer, and see the exact source page used to answer it.

Example questions:

- What was RBI's repo rate decision in February 2025?
- What are Reliance Industries' main business segments?
- What was India's headline inflation in 2024-25?
- What is TCS's primary business?
- How many employees does Infosys have?

## Definition of the Concept

RAG stands for Retrieval-Augmented Generation.

It is an AI architecture where a system first retrieves relevant information from a trusted knowledge base and then uses an LLM to generate an answer from that retrieved context.

In simple words:

1. The user asks a question.
2. The system searches the document collection.
3. It finds the most relevant passages.
4. The LLM reads only those passages.
5. The final answer is generated with source citations.

This is different from directly asking an LLM a question. In normal LLM usage, the model may rely on its training data or make assumptions. In RAG, the model is forced to answer from the documents provided to it.

## Full Forms Used in This Project

- RAG: Retrieval-Augmented Generation
- LLM: Large Language Model
- RBI: Reserve Bank of India
- BSE: Bombay Stock Exchange
- API: Application Programming Interface
- BM25: Best Matching 25, a keyword-based ranking algorithm
- HNSW: Hierarchical Navigable Small World, an approximate nearest-neighbor search method
- PDF: Portable Document Format
- RAGAS: Retrieval-Augmented Generation Assessment, an evaluation framework for RAG systems

## What I Built

I built a production-style Finance RAG system for Q&A over Indian financial documents.

The corpus contains:

- 50 PDF documents
- 16 Indian company annual reports
- 34 RBI publications
- 9,155 pages processed
- 33,229 text chunks
- 33,689 vectors in the vector store

The company annual reports include organizations such as TCS, Infosys, Wipro, HCL Technologies, Reliance Industries, ICICI Bank, Axis Bank, Bajaj Finance, Asian Paints, L&T, Maruti Suzuki, Sun Pharma, ITC, SBI, Kotak Mahindra Bank, and Bharti Airtel.

The RBI documents include annual reports, monetary policy reports, financial stability reports, and economic review chapters.

## Why I Built It

Finance is a domain where accuracy matters.

If someone asks about repo rate, inflation, employee count, revenue, business segments, or financial ratios, the answer should not be guessed. It should come from a trusted source.

This project solves three common problems:

1. Long documents are hard to search manually.
2. Normal LLMs can hallucinate when they do not know the answer.
3. Financial answers need citations, not just fluent text.

The idea was to build a system that behaves like a financial research assistant: fast, source-grounded, and clear about where each answer came from.

## Where This Approach Can Be Used

This kind of system can be used in many real-world domains:

- Finance research over annual reports and regulatory documents
- Banking policy analysis using RBI publications
- Equity research workflows
- Internal company document search
- Legal document Q&A
- Compliance and audit support
- Insurance policy search
- Academic research over large paper collections

Any domain with large trusted documents and high accuracy requirements can benefit from RAG.

## When RAG Is Useful

RAG is useful when:

- The answer must come from specific documents.
- The information changes over time.
- The user needs source citations.
- The dataset is too large to fit directly into one prompt.
- The LLM should not rely only on general training knowledge.

For example, asking "What was RBI's repo rate decision in February 2025?" should be answered from an RBI document, not from memory.

## How the System Works

The pipeline has several stages.

### 1. Document Collection

The system collects RBI publications and Indian company annual reports as PDF files.

The project includes scripts for:

- Downloading RBI documents
- Organizing company annual reports
- Checking corpus inventory
- Validating document counts and structure

### 2. PDF Parsing

PDF parsing is one of the most important parts of this project.

Financial PDFs are not clean plain text files. They contain:

- Headers
- Footers
- Page numbers
- Multi-column layouts
- Tables
- Decorative text
- Cover-page design elements

I used PyMuPDF to extract page-level text and metadata. I also added cleaning logic to remove repeated headers, footers, page numbers, and vertical design text.

### 3. Table-Aware Extraction

This was an important engineering decision.

Financial documents contain many tables. Standard PDF text extraction can lose table meaning. For example, a table like this:

```text
Metric          FY2024    FY2023
Gross NPA (%)    1.24      1.34
Net NPA (%)      0.33      0.35
```

can become plain text without clear labels:

```text
Gross NPA (%) 1.24 1.34 Net NPA (%) 0.33 0.35
```

That makes it harder for the retriever and LLM to understand the data.

To fix this, I added table-aware extraction that tries to preserve table context as labeled text.

### 4. Chunking

After cleaning the documents, the system splits the text into smaller chunks.

Chunking matters because the retriever searches over chunks, not full PDFs. If chunks are too large, they contain noise. If they are too small, they lose context.

I experimented with:

- Recursive chunking
- Token-based chunking
- Sentence-aware chunking

The project uses recursive chunking as the main strategy, with overlap to preserve context between chunks.

### 5. Embeddings

Each chunk is converted into a vector embedding using `BAAI/bge-small-en-v1.5`.

An embedding is a numerical representation of text. Similar meanings are placed closer together in vector space.

This allows the system to understand semantic similarity. For example, a query about "interest rate cut" can still match a passage about "repo rate reduction."

### 6. Vector Store

The embeddings are stored in ChromaDB.

ChromaDB acts as the persistent vector database. It allows the system to search through thousands of chunks quickly and return the most relevant ones.

### 7. Hybrid Retrieval

I used hybrid retrieval: dense vector search plus BM25 keyword search.

This is very important for finance.

Pure vector search is good at meaning, but financial questions often contain exact terms like:

- NPA
- GNPA
- CRR
- repo rate
- inflation
- CAGR
- company names

BM25 helps catch exact keyword matches, while dense retrieval captures semantic meaning.

The final retrieval blend is 70 percent dense search and 30 percent BM25 keyword search.

### 8. Answer Generation

The retrieved chunks are passed to an LLM.

The system prompt tells the model to:

- Answer only from the retrieved context
- Refuse when documents do not contain enough information
- Cite sources inline
- Quote exact numbers for financial data
- Avoid speculation

The current implementation supports OpenAI-compatible providers such as Groq, OpenAI, and local Ollama.

### 9. API and UI

The project includes:

- A FastAPI backend
- A Streamlit chat interface
- Health and stats endpoints
- Query endpoint with optional company and document-type filters

This makes the project usable as both a local app and an API.

## Real Example

Question:

```text
What was RBI's repo rate decision in February 2025?
```

Expected answer style:

```text
The RBI reduced the policy repo rate by 25 basis points to 6.25 per cent in February 2025 [Source: RBI, Page 112].
```

The important part is not only the answer. The important part is that the system also returns the source file, page number, relevance score, and preview of the retrieved passage.

That makes the answer easier to verify.

## Evaluation Results

I evaluated the system using a 15-question test set.

Results:

- Answer rate: 15/15
- Average retrieval relevance: 0.822
- Hallucination rate on the evaluation set: 0 percent
- Total documents: 50
- Total pages processed: 9,155
- Total chunks: 33,229

The test set included questions about RBI policy, inflation, company business models, employee counts, and business segments.

This evaluation helped me compare system quality beyond just "the answer looks good."

## Alternatives to This Approach

RAG is powerful, but it is not the only way to build document Q&A.

### 1. Keyword Search Only

This is the traditional search approach.

Advantages:

- Simple
- Fast
- Easy to explain

Disadvantages:

- Misses semantic meaning
- Poor for natural language questions
- Struggles when the query uses different wording from the document

### 2. Fine-Tuning an LLM

Another option is to fine-tune a model on financial documents.

Advantages:

- Can improve domain-specific style
- Useful for repeated patterns and specialized tasks

Disadvantages:

- Expensive
- Harder to update with new documents
- Does not automatically provide citations
- Can still hallucinate

### 3. Long-Context LLMs

A long-context model can accept large amounts of text directly in the prompt.

Advantages:

- Simpler architecture
- Less retrieval engineering

Disadvantages:

- Can be expensive
- Not ideal for thousands of pages
- Still needs careful source tracking
- Large prompts can add latency

### 4. SQL or Structured Database

If the data is already structured, a database may be better.

Advantages:

- Reliable for exact numbers
- Good for dashboards and analytics
- Easy to query with filters

Disadvantages:

- Requires structured data
- Poor for open-ended document questions
- Does not work well with raw PDFs unless data is extracted first

## Advantages of RAG

- Answers are grounded in trusted documents.
- Source citations improve trust.
- New documents can be added without retraining the model.
- It works well for large document collections.
- It reduces hallucination compared with direct LLM answering.
- It can combine semantic search with keyword search.
- It is practical for real business workflows.

## Disadvantages of RAG

- System quality depends heavily on document parsing.
- Bad chunks lead to bad answers.
- Retrieval errors can still happen.
- Tables in PDFs are difficult to extract correctly.
- Evaluation is more complex than normal software testing.
- Latency can increase because retrieval and generation both happen.
- The pipeline has more moving parts than a simple chatbot.

## Your Learning

### What I Learned

The biggest lesson was that RAG quality is mostly about data quality.

Before building this project, I thought the LLM would be the most important part. After building it, I realized that parsing, cleaning, chunking, metadata, and retrieval design decide most of the final answer quality.

I also learned:

- Hybrid retrieval works better than pure vector search for finance.
- Metadata tagging is critical for filtering by company or document type.
- Source citations make AI answers more trustworthy.
- RAG evaluation should be measurable, not only based on manual checking.
- A small model can perform well if retrieval quality is strong.

### Mistakes I Made

One mistake was assuming that all downloaded PDFs were the correct documents.

During corpus validation, I found that one wrong document could affect the whole evaluation. For example, downloading a related financial document instead of the correct company annual report can make the system fail specific company questions even if the retriever score looks high.

Another mistake was trusting normal PDF text extraction too much.

For financial documents, tables are everywhere. Basic extraction often removes the relationship between headers and values. That can make numbers meaningless.

I also saw that pure vector search can miss exact financial acronyms. Terms like NPA, GNPA, CRR, and repo rate need keyword matching support.

### How I Fixed Them

I added corpus validation scripts to check the document collection instead of only trusting file counts.

I improved PDF parsing by removing repeated noise such as headers, footers, page numbers, and vertical design text.

I added table-aware extraction so financial values keep more context.

I switched to hybrid retrieval, combining dense embeddings with BM25 keyword scoring.

I also added evaluation using a fixed test set so improvements could be measured instead of guessed.

## Final Thoughts

This project taught me that building a useful AI system is not just about calling an LLM API.

The real engineering is in the pipeline:

- collecting the right documents
- cleaning noisy PDFs
- splitting text correctly
- retrieving the right chunks
- grounding the LLM
- evaluating the output
- exposing it through an API and UI

RAG is a practical approach for building trustworthy AI applications over private or domain-specific knowledge.

For finance, where every number and citation matters, RAG is not just a nice architecture. It is often the difference between a chatbot that sounds confident and a system that can actually be trusted.

