#!/usr/bin/env python
"""Full RAGAS evaluation using a Groq-hosted model as the judge LLM.

Metrics (ragas 0.2.x):
  faithfulness        – answer is supported by the retrieved context
  context_precision   – relevant chunks ranked high in retrieval
  context_recall      – retrieved context covers the ground truth
  answer_relevancy    – answer addresses the question

The judge LLM is a Groq model accessed via the OpenAI-compatible endpoint
(no OpenAI credits needed).  Set RAGAS_JUDGE_MODEL in .env to switch models;
default is ``llama-3.1-8b-instant``.

Usage::

    python scripts/ragas_eval.py \\
        --eval-set data/test_sets/eval_set_v2.jsonl \\
        --base-url http://localhost:8000 \\
        --out data/test_sets/ragas_results.json

Requirements (already in venv):
    ragas>=0.2.0, langchain-openai, requests

Rows skipped:
  - skip=true (template rows)
  - expect.refuse=true (refusal rows – no ground_truth, meaningless to score)
  - no contains_all (no ground_truth to supply context_recall)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Project root on sys.path so src.* imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_JUDGE_MODEL = "llama-3.1-8b-instant"
DEFAULT_REFUSAL_RE = r"do(?:es)? not contain sufficient information|insufficient information"


# ─────────────────────────────────────────── helpers ──────────────────────────

def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{i}: invalid JSON ({exc})")
            rows.append(row)
    return rows


def query_api(
    session: requests.Session,
    base_url: str,
    row: dict,
    top_k: int,
    api_key: Optional[str],
    retrieval_mode: str = "hybrid",
    timeout: float = 60,
) -> Optional[dict]:
    body: dict[str, Any] = {
        "question": row["question"],
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
    }
    for k in ("company", "doc_type"):
        if row.get(k):
            body[k] = row[k]
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = session.post(f"{base_url}/query", json=body, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        logger.warning("API returned %d for row %s", r.status_code, row.get("id"))
    except requests.RequestException as exc:
        logger.warning("Request failed for row %s: %s", row.get("id"), exc)
    return None


def build_ragas_dataset(
    rows: list[dict],
    responses: list[Optional[dict]],
) -> tuple[list[dict], int]:
    """
    Convert eval-set rows + API responses to RAGAS Dataset records.

    Returns (records, skipped_count).
    Only rows with `contains_all` (ground truth) and a valid API response
    are included; refusal rows and no-ground-truth rows are skipped.
    """
    records: list[dict] = []
    skipped = 0
    for row, resp in zip(rows, responses):
        if resp is None:
            skipped += 1
            continue
        exp = row.get("expect", {})
        if exp.get("refuse"):
            skipped += 1  # no meaningful ground_truth for refusal rows
            continue
        ground_truths = exp.get("contains_all", [])
        if not ground_truths:
            skipped += 1  # no ground_truth → context_recall undefined
            continue

        answer = resp.get("answer", "") or ""
        contexts = [s.get("preview", "") for s in (resp.get("sources") or [])]
        if not contexts:
            skipped += 1
            continue

        records.append({
            "question": row["question"],
            "answer": answer,
            "contexts": contexts,
            # RAGAS expects a single ground_truth string
            "ground_truth": " ".join(ground_truths),
        })
    return records, skipped


# ─────────────────────────────────────────── RAGAS evaluation ─────────────────

def build_groq_llm(judge_model: str, groq_api_key: str):
    """
    Return a ragas-compatible LLM wrapper backed by a Groq-hosted model.

    Strategy: use langchain_openai.ChatOpenAI pointed at Groq's OpenAI-
    compatible endpoint, then wrap it with ragas.llms.LangchainLLMWrapper.
    This avoids touching the OPENAI_API_KEY env var entirely.
    """
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Install with: pip install langchain-openai ragas"
        )

    chat_model = ChatOpenAI(
        model=judge_model,
        api_key=groq_api_key,  # type: ignore[arg-type]
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat_model)


def build_groq_embeddings(groq_api_key: str):
    """
    Return an embeddings wrapper for RAGAS answer_relevancy metric.

    RAGAS answer_relevancy requires an embeddings model.  We use the
    sentence-transformers model already present in the project venv so no
    extra API key is needed.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
    except ImportError:
        # Gracefully degrade: return None, answer_relevancy will be skipped
        logger.warning(
            "langchain-community not installed; answer_relevancy will be skipped."
        )
        return None

    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )
    return LangchainEmbeddingsWrapper(embed_model)


def run_ragas(
    records: list[dict],
    judge_model: str,
    groq_api_key: str,
    batch_size: int = 5,
) -> dict:
    """
    Run RAGAS metrics on `records` and return a dict of averaged scores.

    ragas 0.2.x API::

        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, ...
        dataset = Dataset.from_list(records)
        result = evaluate(dataset, metrics=[...], llm=llm)
    """
    try:
        from datasets import Dataset  # noqa: PLC0415
        from ragas import evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Install with: pip install ragas datasets"
        )

    llm_wrapper = build_groq_llm(judge_model, groq_api_key)
    emb_wrapper = build_groq_embeddings(groq_api_key)

    # Instantiate metrics with the Groq judge
    faithfulness_metric = Faithfulness(llm=llm_wrapper)
    context_precision_metric = ContextPrecision(llm=llm_wrapper)
    context_recall_metric = ContextRecall(llm=llm_wrapper)
    metrics = [faithfulness_metric, context_precision_metric, context_recall_metric]

    if emb_wrapper is not None:
        answer_relevancy_metric = AnswerRelevancy(llm=llm_wrapper, embeddings=emb_wrapper)
        metrics.append(answer_relevancy_metric)
    else:
        logger.warning("Skipping answer_relevancy (no embeddings wrapper).")

    dataset = Dataset.from_list(records)
    logger.info("Running RAGAS on %d rows with judge=%s …", len(records), judge_model)
    t0 = time.perf_counter()
    result = evaluate(dataset, metrics=metrics)
    elapsed = round((time.perf_counter() - t0), 1)
    logger.info("RAGAS evaluation completed in %.1fs", elapsed)

    # Convert to plain dict; ragas ≥0.2 returns a pandas-backed EvaluationResult
    try:
        scores_df = result.to_pandas()
        per_row = scores_df.to_dict(orient="records")
        averages: dict[str, float] = {}
        for col in scores_df.columns:
            if col in ("question", "answer", "contexts", "ground_truth"):
                continue
            try:
                averages[col] = round(float(scores_df[col].mean()), 4)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        # Fallback for older ragas API
        per_row = []
        averages = {k: round(float(v), 4) for k, v in result.items()}

    return {"averages": averages, "per_row": per_row, "elapsed_s": elapsed}


# ─────────────────────────────────────────── CLI ──────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-set", default="data/test_sets/eval_set_v2.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--retrieval-mode", default="hybrid",
                    choices=["dense", "hybrid", "rerank"])
    ap.add_argument("--api-key", default=os.getenv("EVAL_API_KEY"))
    ap.add_argument(
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help=f"Groq model to use as RAGAS judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    ap.add_argument(
        "--groq-api-key",
        default=os.getenv("GROQ_API_KEY", ""),
        help="Groq API key (defaults to GROQ_API_KEY env var)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    if not args.groq_api_key:
        sys.exit(
            "GROQ_API_KEY is not set. Export it or pass --groq-api-key.\n"
            "Get a free key at https://console.groq.com"
        )

    rows = load_rows(args.eval_set)
    logger.info("Loaded %d rows from %s", len(rows), args.eval_set)

    # ── Collect API responses ─────────────────────────────────────────────── #
    responses: list[Optional[dict]] = []
    with requests.Session() as session:
        for row in rows:
            if row.get("skip"):
                responses.append(None)
                continue
            resp = query_api(
                session, args.base_url, row,
                top_k=args.top_k,
                api_key=args.api_key,
                retrieval_mode=args.retrieval_mode,
                timeout=args.timeout,
            )
            responses.append(resp)
            time.sleep(0.2)  # be gentle to the API

    # ── Build RAGAS dataset ───────────────────────────────────────────────── #
    records, skipped = build_ragas_dataset(rows, responses)
    logger.info(
        "RAGAS dataset: %d scoreable rows, %d skipped (refusals / no ground_truth / API errors)",
        len(records), skipped,
    )

    if not records:
        sys.exit(
            "No scoreable rows found.\n"
            "Make sure the API is running and eval_set_v2.jsonl has rows with contains_all."
        )

    # ── Run RAGAS ─────────────────────────────────────────────────────────── #
    ragas_result = run_ragas(records, args.judge_model, args.groq_api_key)

    # ── Print summary ─────────────────────────────────────────────────────── #
    print("\n── RAGAS scores ──────────────────────────────────────────")
    for metric, score in ragas_result["averages"].items():
        print(f"  {metric:<30} {score:.4f}")
    print(f"\n  Evaluated {len(records)} rows in {ragas_result['elapsed_s']}s")

    # ── Save report ───────────────────────────────────────────────────────── #
    out = args.out or f"data/test_sets/ragas_{datetime.now():%Y%m%d_%H%M}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "base_url": args.base_url,
                "eval_set": args.eval_set,
                "judge_model": args.judge_model,
                "retrieval_mode": args.retrieval_mode,
                "rows_scored": len(records),
                "rows_skipped": skipped,
                "averages": ragas_result["averages"],
                "elapsed_s": ragas_result["elapsed_s"],
                "per_row": ragas_result["per_row"],
            },
            indent=2,
            ensure_ascii=False,
            default=str,  # handle any non-serialisable ragas objects
        ),
        encoding="utf-8",
    )
    print(f"\nReport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
