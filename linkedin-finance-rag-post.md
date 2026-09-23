I built a Finance RAG system for Indian financial documents.

Not a normal chatbot.

This system can search through company annual reports and RBI publications, then answer questions with page-level source citations.

The surprising part?

The LLM was not the hardest part.

The real challenge was:

- messy PDFs
- broken table extraction
- financial acronyms like NPA, CRR, GNPA
- noisy headers and footers
- wrong document downloads
- making sure the model answers from sources, not memory

What the system handles:

- 50 financial documents
- 9,155 pages processed
- 33,000+ chunks
- hybrid retrieval using embeddings + BM25
- FastAPI backend
- Streamlit chat UI
- source-cited answers

One big lesson:

RAG is not just about connecting an LLM to documents.

It is about building a reliable information pipeline.

If retrieval fails, the answer fails.
If parsing is weak, the answer is weak.
If citations are missing, trust is missing.

I also created an architecture diagram to show how the pipeline works from PDFs to final answers.

GitHub: https://github.com/preetham789/finance-rag
Blog: [Add your Hashnode blog link here]

Would love feedback from people building with RAG, LLMs, or document AI.

#AI #RAG #LLM #MachineLearning #Python #FastAPI #Streamlit #VectorDatabase #Finance #GenerativeAI #OpenToWork
