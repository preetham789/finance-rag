# scripts/query.py
r"""
Query the RAG system from the command line.

Usage:
  python scripts\query.py --question "What was RBI's inflation target in 2024?"
  python scripts\query.py --question "What is HDFC Bank's NPA ratio?" --company "HDFC Bank"
  python scripts\query.py --question "Compare TCS and Infosys revenue" --verbose
  python scripts\query.py --interactive
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import CHROMA_DIR, GROQ_API_KEY, OPENAI_API_KEY
from src.ingestion.embedder import (
    EMBEDDING_MODEL_NAME,
    build_chroma_client, get_or_create_collection,
)
from src.retrieval.retriever import FinanceRetriever
from src.generation.chain import FinanceRAGChain, LLMGenerationError
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during queries


def build_chain(top_k: int = 5) -> FinanceRAGChain:
    """Load ChromaDB, embedding model, and build the full RAG chain."""
    print("Loading RAG system...", end=" ", flush=True)

    client     = build_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        print("\nERROR: ChromaDB is empty. Run: python scripts\\embed.py first.")
        sys.exit(1)

    model     = SentenceTransformer(EMBEDDING_MODEL_NAME)
    retriever = FinanceRetriever(collection, model, top_k=top_k)

    # ── Auto-detect which provider to use based on .env ──
    if GROQ_API_KEY:
        chain = FinanceRAGChain(retriever, api_key=GROQ_API_KEY, provider="groq")
        provider_info = "Groq / llama-3.1-8b-instant (free)"
    elif OPENAI_API_KEY:
        chain = FinanceRAGChain(
            retriever,
            api_key=OPENAI_API_KEY,
            provider="openai",
        )
        provider_info = "OpenAI / gpt-4o-mini"
    else:
        print("\nERROR: No API key found. Add GROQ_API_KEY or OPENAI_API_KEY to .env")
        sys.exit(1)

    print(f"ready. ({collection.count():,} vectors | {provider_info})\n")
    return chain


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print a query result to the terminal."""
    print("=" * 60)
    print(f"Q: {result['question']}")
    print("=" * 60)
    print(f"\n{result['answer']}\n")

    print("-" * 60)
    print(f"Sources ({len(result['sources'])} retrieved):")
    for i, src in enumerate(result["sources"], 1):
        score = src.get("score", 0)
        print(f"  [{i}] {src['company']}  |  Page {src['page_number']}  "
              f"|  relevance {score:.3f}")
        if verbose:
            print(f"       {src['preview']}")

    print(f"\nModel: {result['model']}  |  "
          f"Tokens: {result.get('tokens_used', '?')}  |  "
          f"Mode: {result['retrieval_mode']}")
    print("=" * 60)


def run_query(
    chain: FinanceRAGChain,
    question: str,
    where_filter: dict | None = None,
    verbose: bool = False,
) -> bool:
    """Run a query and render either a normal answer or a friendly API error."""
    try:
        result = chain.query(
            question=question,
            where_filter=where_filter,
            verbose=verbose,
        )
    except LLMGenerationError as exc:
        print_result(exc.as_result(), verbose=verbose)
        return False

    print_result(result, verbose=verbose)
    return True


def interactive_mode(chain: FinanceRAGChain) -> None:
    """Run an interactive Q&A loop in the terminal."""
    print("Finance RAG — Interactive Mode")
    print("Commands:  'quit' to exit  |  'filter <company>' to restrict search")
    print("           'clear filter' to remove filter  |  'verbose' to toggle details")
    print("-" * 60)

    where_filter = None
    verbose      = False

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}")
            continue
        if user_input.lower().startswith("filter "):
            company = user_input[7:].strip()
            where_filter = {"company": company}
            print(f"Filter set: company = '{company}'")
            continue
        if user_input.lower() == "clear filter":
            where_filter = None
            print("Filter cleared.")
            continue

        run_query(
            chain,
            question=user_input,
            where_filter=where_filter,
            verbose=verbose,
        )


def main():
    parser = argparse.ArgumentParser(description="Query the Finance RAG system")
    parser.add_argument("--question",     type=str, help="Question to ask")
    parser.add_argument("--company",      type=str, help="Filter to a specific company")
    parser.add_argument("--doc-type",     type=str, choices=["rbi", "company"],
                        help="Filter to doc type")
    parser.add_argument("--top-k",        type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Show retrieved chunk previews")
    parser.add_argument("--interactive",  action="store_true",
                        help="Start interactive Q&A session")
    args = parser.parse_args()

    chain = build_chain(top_k=args.top_k)

    # Build where filter from CLI args
    where_filter = None
    if args.company:
        where_filter = {"company": args.company}
    elif args.doc_type:
        where_filter = {"doc_type": args.doc_type}

    if args.interactive:
        interactive_mode(chain)

    elif args.question:
        success = run_query(
            chain,
            question=args.question,
            where_filter=where_filter,
            verbose=args.verbose,
        )
        if not success:
            sys.exit(1)

    else:
        # Demo mode — run 5 built-in test questions
        print("No question provided — running demo queries\n")
        demo_questions = [
            ("What was RBI's inflation target in FY2024?",
             {"doc_type": "rbi"}),
            ("What is HDFC Bank's gross NPA ratio?",
             {"company": "HDFC Bank"}),
            ("What was TCS's total revenue in FY2024?",
             {"company": "TCS"}),
            ("What are the key risks mentioned by Reliance Industries?",
             {"company": "Reliance"}),
            ("What was the repo rate decision by RBI in 2024?",
             {"doc_type": "rbi"}),
        ]
        for question, where in demo_questions:
            success = run_query(chain, question=question, where_filter=where)
            if not success:
                print("\nStopping demo queries because answer generation is unavailable.")
                sys.exit(1)
            print()


if __name__ == "__main__":
    main()
