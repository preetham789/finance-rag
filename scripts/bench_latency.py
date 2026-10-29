#!/usr/bin/env python
"""Latency + concurrency benchmark against a running Finance RAG API.

  python scripts/bench_latency.py --concurrency 1,2,4,8 --n 24
  python scripts/bench_latency.py --questions-file data/test_sets/eval_set_v2.jsonl

Reports client-side p50/p95/p99, throughput and error mix per concurrency level.
429s and timeouts are counted, not hidden: that is where the system actually breaks
(for this project, expect the Groq rate limit before the vector store).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Optional

import requests

DEFAULT_QUESTIONS = [
    "What was RBI's repo rate in February 2025?",
    "What is TCS's primary business?",
    "What are Reliance's main business segments?",
    "What was India's inflation in 2024-25?",
    "How many employees does Infosys have?",
]


def percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    return sorted_vals[max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)]


def _ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.0f}"


def one_call(
    base_url: str, question: str, api_key: Optional[str], timeout: float
) -> tuple[str, float, Optional[float]]:
    t0 = time.perf_counter()
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = requests.post(
            f"{base_url}/query",
            json={"question": question},
            headers=headers,
            timeout=timeout,
        )
        status = str(r.status_code)
        server = r.json().get("latency_ms") if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        status, server = "timeout/conn", None
    return status, (time.perf_counter() - t0) * 1000, server


def run_level(
    base_url: str,
    questions: list[str],
    n: int,
    concurrency: int,
    api_key: Optional[str] = None,
    timeout: float = 60,
) -> dict:
    qs = [q for _, q in zip(range(n), cycle(questions))]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda q: one_call(base_url, q, api_key, timeout), qs))
    wall = time.perf_counter() - t0
    ok = [r for r in results if r[0] == "200"]
    lat = sorted(r[1] for r in ok)
    server = [r[2] for r in ok if r[2] is not None]
    errors: dict[str, int] = {}
    for status, _, _ in results:
        if status != "200":
            errors[status] = errors.get(status, 0) + 1
    return {
        "concurrency": concurrency,
        "n": n,
        "ok": len(ok),
        "errors": errors,
        "rps": round(len(ok) / wall, 2) if wall else None,
        "client_p50_ms": percentile(lat, 50),
        "client_p95_ms": percentile(lat, 95),
        "client_p99_ms": percentile(lat, 99),
        "server_mean_ms": round(statistics.mean(server)) if server else None,
    }


def load_questions(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                row = json.loads(line)
                if not row.get("skip") and not row.get("expect", {}).get("refuse"):
                    out.append(row["question"])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=24, help="requests per concurrency level")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--questions-file", default=None)
    ap.add_argument("--api-key", default=os.getenv("EVAL_API_KEY"))
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    questions = (
        load_questions(args.questions_file) if args.questions_file else DEFAULT_QUESTIONS
    )
    for q in questions[:2]:  # warm-up, not measured
        one_call(args.base_url, q, args.api_key, args.timeout)

    results = []
    hdr = f"{'conc':>5}{'ok/n':>8}{'rps':>7}{'p50':>8}{'p95':>8}{'p99':>8}{'srv_mean':>10}  errors"
    print(hdr)
    for level in (int(x) for x in args.concurrency.split(",")):
        res = run_level(
            args.base_url, questions, args.n, level, args.api_key, args.timeout
        )
        results.append(res)
        print(
            f"{res['concurrency']:>5}{str(res['ok']) + '/' + str(res['n']):>8}"
            f"{res['rps']:>7}{_ms(res['client_p50_ms']):>8}{_ms(res['client_p95_ms']):>8}"
            f"{_ms(res['client_p99_ms']):>8}{_ms(res['server_mean_ms']):>10}  {res['errors'] or ''}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
