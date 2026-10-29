# src/evaluation/__init__.py
"""
Evaluation package for Finance RAG.
"""
from src.evaluation.dataset import (
    EVAL_DATASET,
    export_to_jsonl,
    get_eval_dataset,
    load_from_jsonl,
    to_ragas_format,
)
from src.evaluation.ragas_eval import (
    TEST_DATASET,
    build_ragas_dataset,
    print_ragas_report,
)

__all__ = [
    "EVAL_DATASET",
    "TEST_DATASET",
    "get_eval_dataset",
    "to_ragas_format",
    "export_to_jsonl",
    "load_from_jsonl",
    "build_ragas_dataset",
    "print_ragas_report",
]
