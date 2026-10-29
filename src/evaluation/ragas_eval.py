# src/evaluation/ragas_eval.py
"""
RAGAS evaluation module for Finance RAG.

Provides utilities to evaluate the RAG pipeline using RAGAS metrics
(faithfulness, answer_relevancy, context_precision, context_recall)
against the curated financial QA dataset.
"""
from __future__ import annotations

import logging
from typing import Any

from src.evaluation.dataset import EVAL_DATASET, to_ragas_format

logger = logging.getLogger(__name__)

# Full RAGAS-compatible dataset generated dynamically from the curated EVAL_DATASET
TEST_DATASET: list[dict[str, Any]] = to_ragas_format(EVAL_DATASET)


def build_ragas_dataset(chain, test_data: list[dict[str, Any]], verbose: bool = True) -> list[dict[str, Any]]:
    results = []
    print(f"Running {len(test_data)} questions...\n")

    for i, item in enumerate(test_data, 1):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")
        where_filter = item.get("filter")

        if verbose:
            print(f"[{i:02d}/{len(test_data)}] {question[:65]}")

        try:
            result   = chain.query(question=question, where_filter=where_filter)
            contexts = [c["text"] for c in result.get("chunks", [])]
            answered = "insufficient information" not in result["answer"].lower()

            results.append({
                "question":     question,
                "answer":       result["answer"],
                "contexts":     contexts,
                "ground_truth": ground_truth,
                "filter":       str(where_filter),
                "top_score":    result["chunks"][0].get("hybrid_score", 0) if result.get("chunks") else 0,
                "tokens_used":  result.get("tokens_used", 0),
            })

            if verbose:
                print(f"         [{'OK' if answered else 'NO'}]  score={results[-1]['top_score']:.3f}")

        except Exception as e:
            logger.error(f"Q{i} failed: {e}")
            results.append({
                "question":     question,
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": ground_truth,
                "filter":       str(where_filter),
                "top_score":    0,
                "tokens_used":  0,
            })

    return results


def print_ragas_report(scores: dict[str, float], ragas_dataset: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION REPORT")
    print("=" * 60)
    thresholds = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_precision": 0.70,
        "context_recall": 0.65,
    }
    for metric, threshold in thresholds.items():
        score  = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "NEEDS WORK"
        bar    = "█" * int(score * 20)
        print(f"\n  {metric}")
        print(f"  {score:.3f}  [{bar:<20}]  {status}")
    print("=" * 60)
