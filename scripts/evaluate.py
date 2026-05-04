# scripts/evaluate.py
import sys, json, argparse, os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-only", action="store_true")
    parser.add_argument("--save",         action="store_true")
    parser.add_argument("--questions",    type=int, default=len(TEST_DATASET))
    args = parser.parse_args()

    print("=" * 60)
    print("Finance RAG — Evaluation")
    print("=" * 60)

    chain     = build_chain()
    test_data = TEST_DATASET[:args.questions]
    results   = build_ragas_dataset(chain, test_data, verbose=True)

    # Answer rate
    answered = sum(1 for d in results
                   if "insufficient information" not in d["answer"].lower()
                   and not d["answer"].startswith("ERROR"))
    total = len(results)

    print(f"\n{'='*60}")
    print(f"Answer rate: {answered}/{total} ({100*answered//total}%)")
    print(f"{'='*60}")

    # Print each Q&A
    for i, d in enumerate(results, 1):
        ok = "insufficient information" not in d["answer"].lower()
        print(f"\n[{'OK' if ok else 'NO'}] Q{i:02d}: {d['question']}")
        print(f"      score={d.get('top_score',0):.3f}")
        print(f"      A: {d['answer'][:200]}")

    if args.save:
        TEST_SETS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M")
        path = TEST_SETS_DIR / f"eval_{ts}.json"
        path.write_text(json.dumps({"timestamp": ts, "answer_rate": answered/total,
                                    "dataset": results}, indent=2, ensure_ascii=True))
        print(f"\nSaved → {path}")

if __name__ == "__main__":
    main()