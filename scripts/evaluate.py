# scripts/evaluate.py
"""
Evaluate the Finance RAG system.

Runs 15 built-in test questions and reports:
  1. Answer rate  — how many questions got a non-fallback answer
  2. RAGAS metrics (if --ragas flag is passed):
       - faithfulness       : does the answer only use retrieved context?
       - answer_relevancy   : does the answer address the question?
       - context_precision  : are retrieved chunks useful?
       - context_recall     : does context contain what's needed?

Usage:
  python scripts\\evaluate.py                    # fast: answer rate only
  python scripts\\evaluate.py --ragas            # full RAGAS metrics (needs OPENAI_API_KEY)
  python scripts\\evaluate.py --questions 5      # run only first 5 questions
  python scripts\\evaluate.py --save             # save results to data/test_sets/
"""
import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.config import CHROMA_DIR, TEST_SETS_DIR
from src.ingestion.embedder import EMBEDDING_MODEL_NAME, COLLECTION_NAME, build_chroma_client, get_or_create_collection
from src.retrieval.retriever import FinanceRetriever
from src.generation.chain import FinanceRAGChain
from src.evaluation.ragas_eval import TEST_DATASET, build_ragas_dataset, print_ragas_report
from sentence_transformers import SentenceTransformer


def build_chain():
    client     = build_chroma_client(CHROMA_DIR)
    collection = get_or_create_collection(client)
    model      = SentenceTransformer(EMBEDDING_MODEL_NAME)
    retriever  = FinanceRetriever(collection, model, top_k=5)
    groq_key   = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if groq_key:
        return FinanceRAGChain(retriever, api_key=groq_key, provider="groq")
    elif openai_key:
        return FinanceRAGChain(retriever, api_key=openai_key, provider="openai")
    else:
        print("ERROR: No API key found in .env"); sys.exit(1)


def run_answer_rate(chain, test_data: list, verbose: bool = False) -> list:
    """Run all test questions and print per-question results."""
    results = build_ragas_dataset(chain, test_data, verbose=True)

    answered = sum(
        1 for d in results
        if "insufficient information" not in d["answer"].lower()
        and not d["answer"].startswith("ERROR")
    )
    total = len(results)

    print(f"\n{'='*60}")
    print(f"Answer rate: {answered}/{total} ({100*answered/total:.1f}%)")   # N8 FIX: was //
    print(f"{'='*60}")

    for i, d in enumerate(results, 1):
        ok = "insufficient information" not in d["answer"].lower()
        print(f"\n[{'OK' if ok else 'NO'}] Q{i:02d}: {d['question']}")
        print(f"      score={d.get('top_score', 0):.3f}")
        print(f"      A: {d['answer'][:200]}")

    return results


def run_ragas_metrics(results: list) -> dict:
    """
    N2 FIX: Actually compute RAGAS metrics — was imported but never called.

    Requires:
      pip install ragas
      OPENAI_API_KEY set (RAGAS uses GPT-4 as an LLM judge by default)

    Metrics computed:
      faithfulness      — fraction of answer claims grounded in retrieved context
      answer_relevancy  — how well the answer addresses the question
      context_precision — fraction of retrieved chunks that are relevant
      context_recall    — fraction of ground truth covered by retrieved chunks
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        print("\nRAGAS not installed. Run: pip install ragas datasets")
        return {}

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("\nRAGAS requires OPENAI_API_KEY (uses GPT-4 as judge). Skipping RAGAS metrics.")
        return {}

    # Build the HuggingFace Dataset that RAGAS expects
    ragas_rows = []
    for d in results:
        if not d.get("contexts"):
            continue
        ragas_rows.append({
            "question":  d["question"],
            "answer":    d["answer"],
            "contexts":  d["contexts"],          # list[str] — retrieved chunk texts
            "ground_truth": d.get("ground_truth", ""),
        })

    if not ragas_rows:
        print("No rows have 'contexts' — make sure build_ragas_dataset returns context texts.")
        return {}

    dataset = Dataset.from_list(ragas_rows)

    print(f"\nRunning RAGAS metrics on {len(ragas_rows)} questions...")
    print("(This calls OpenAI GPT-4 as a judge — may take 1-2 minutes)\n")

    try:
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        print_ragas_report(scores, results)
        return scores
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Finance RAG")
    parser.add_argument("--ragas",        action="store_true",
                        help="Run full RAGAS metrics (requires OPENAI_API_KEY)")
    parser.add_argument("--answers-only", action="store_true",
                        help="Skip sources, show answers only")
    parser.add_argument("--save",         action="store_true",
                        help="Save results to data/test_sets/")
    parser.add_argument("--questions",    type=int, default=len(TEST_DATASET),
                        help=f"Number of test questions to run (default: {len(TEST_DATASET)})")
    args = parser.parse_args()

    print("=" * 60)
    print("Finance RAG — Evaluation")
    print("=" * 60)

    chain     = build_chain()
    test_data = TEST_DATASET[:args.questions]

    # ── Step 1: Answer rate (fast, no external API) ──
    results = run_answer_rate(chain, test_data, verbose=not args.answers_only)

    # ── Step 2: RAGAS metrics (optional, needs OpenAI) ──
    if args.ragas:
        ragas_scores = run_ragas_metrics(results)
    else:
        print("\nTip: run with --ragas to compute faithfulness/relevancy metrics")
        print("     (requires OPENAI_API_KEY as RAGAS uses GPT-4 as a judge)")

    # ── Step 3: Save ──
    if args.save:
        TEST_SETS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M")
        path = TEST_SETS_DIR / f"eval_{ts}.json"
        path.write_text(
            json.dumps(
                {
                    "timestamp":   ts,
                    "answer_rate": sum(
                        1 for d in results
                        if "insufficient information" not in d["answer"].lower()
                    ) / len(results),
                    "dataset": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()