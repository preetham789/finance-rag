#!/usr/bin/env python
"""API-level evaluation for the Finance RAG service.

Deterministic metrics: no judge LLM, no API credits, same input -> same output.

  hit_rate / mrr       expected (company, page) appeared in the returned sources
  answer_ok_rate       required strings present in the answer (case/comma-insensitive)
  refusal_accuracy     should-refuse questions (out of corpus, traps) that were refused
  false_refusal_rate   answerable questions that were wrongly refused
  citation_rate        answerable, non-refused answers carrying a [Source: ...] tag

Run:      python scripts/eval_api.py --eval-set data/test_sets/eval_set_v2.jsonl
Compare:  python scripts/eval_api.py --compare old.json new.json   (exit 1 on regression)
A/B:      python scripts/eval_api.py --ab-sweep --alphas 0.3,0.5,0.7,0.9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

DEFAULT_REFUSAL = (
    r"do(?:es)? not contain sufficient information|insufficient information"
)
CITATION_RE = re.compile(r"\[Source:[^\]]+\]", re.I)
GROUPED_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{2,3})+(?!\d)")
HIGHER_IS_BETTER = ("hit_rate", "mrr", "answer_ok_rate", "refusal_accuracy")


# ---------------------------------------------------------------- matching ----
def norm_text(s: str) -> str:
    s = GROUPED_NUM_RE.sub(lambda m: m.group(0).replace(",", ""), s or "")
    return re.sub(r"\s+", " ", s.lower()).strip()


def _co(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def company_eq(a: Any, b: Any) -> bool:
    x, y = _co(a), _co(b)
    return bool(x and y and (x in y or y in x))


def source_matches(src: dict, exp: dict, tol: int = 0) -> bool:
    if exp.get("company") and not company_eq(src.get("company"), exp["company"]):
        return False
    pages = exp.get("pages") or []
    if not pages:
        return True
    p = src.get("page_number")
    return p is not None and any(abs(int(p) - int(x)) <= tol for x in pages)


def first_hit_rank(
    sources: list[dict], expected: list[dict], tol: int = 0
) -> Optional[int]:
    for rank, src in enumerate(sources, start=1):
        if any(source_matches(src, e, tol) for e in expected):
            return rank
    return None


# ----------------------------------------------------------------- scoring ----
def score_row(
    row: dict, resp: Optional[dict], refusal_re: re.Pattern, tol: int = 0
) -> dict:
    exp = row.get("expect", {})
    should_refuse = bool(exp.get("refuse"))
    out: dict[str, Any] = {
        "id": row["id"],
        "category": row.get("category", "uncategorised"),
        "should_refuse": should_refuse,
        "error": None,
    }
    if resp is None:
        out["error"] = "no valid response"
        return out

    answer = resp.get("answer", "") or ""
    sources = resp.get("sources", []) or []
    refused = bool(refusal_re.search(answer))
    out["refused"] = refused
    out["server_latency_ms"] = resp.get("latency_ms")

    exp_src = exp.get("sources") or []
    if exp_src and not should_refuse:
        rank = first_hit_rank(sources, exp_src, tol)
        out["hit"], out["rank"] = rank is not None, rank

    if should_refuse:
        out["refusal_correct"] = refused
    else:
        na = norm_text(answer)
        all_ok = all(norm_text(s) in na for s in exp.get("contains_all", []))
        any_list = exp.get("contains_any", [])
        any_ok = (not any_list) or any(norm_text(s) in na for s in any_list)
        out["answer_ok"] = (not refused) and all_ok and any_ok
        out["false_refusal"] = refused
        out["cited"] = bool(CITATION_RE.search(answer))

    out["answer"] = answer[:400]
    out["top_sources"] = [(s.get("company"), s.get("page_number")) for s in sources[:5]]
    return out


def percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    return sorted_vals[max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)]


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def aggregate(scored: list[dict]) -> dict:
    ok = [r for r in scored if not r.get("error")]
    retr = [r for r in ok if "hit" in r]
    ans = [r for r in ok if not r["should_refuse"]]
    cited_pool = [r for r in ans if not r["refused"]]
    ref = [r for r in ok if r["should_refuse"]]
    lat = sorted(
        r["server_latency_ms"] for r in ok if r.get("server_latency_ms") is not None
    )
    return {
        "n": len(scored),
        "errors": len(scored) - len(ok),
        "retrieval_n": len(retr),
        "hit_rate": _rate(sum(r["hit"] for r in retr), len(retr)),
        "mrr": (
            round(sum(1 / r["rank"] if r["rank"] else 0 for r in retr) / len(retr), 4)
            if retr
            else None
        ),
        "answerable_n": len(ans),
        "answer_ok_rate": _rate(sum(r["answer_ok"] for r in ans), len(ans)),
        "false_refusal_rate": _rate(sum(r["false_refusal"] for r in ans), len(ans)),
        "citation_rate": _rate(sum(r["cited"] for r in cited_pool), len(cited_pool)),
        "refusal_n": len(ref),
        "refusal_accuracy": _rate(sum(r["refusal_correct"] for r in ref), len(ref)),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
    }


# -------------------------------------------------------------------- I/O ----
def query_api(
    session: requests.Session,
    base_url: str,
    row: dict,
    top_k: int,
    api_key: Optional[str],
    timeout: float = 60,
    retries: int = 3,
    retrieval_mode: str = "hybrid",
    alpha: Optional[float] = None,
) -> Optional[dict]:
    body: dict[str, Any] = {
        "question": row["question"],
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
    }
    if alpha is not None:
        body["alpha"] = alpha
    for k in ("company", "doc_type"):
        if row.get(k):
            body[k] = row[k]
    headers = {"X-API-Key": api_key} if api_key else {}
    for attempt in range(retries + 1):
        wait = 2.0**attempt
        try:
            r = session.post(
                f"{base_url}/query", json=body, headers=headers, timeout=timeout
            )
        except requests.RequestException:
            r = None
        if r is not None:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                return None
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", wait))
        if attempt < retries:
            time.sleep(min(wait, 15))
    return None


def run_eval(
    base_url: str,
    rows: list[dict],
    *,
    top_k: int = 5,
    api_key: Optional[str] = None,
    tol: int = 0,
    sleep: float = 0.0,
    refusal_pattern: str = DEFAULT_REFUSAL,
    timeout: float = 60,
    retries: int = 3,
    retrieval_mode: str = "hybrid",
    alpha: Optional[float] = None,
) -> tuple[list[dict], int]:
    refusal_re = re.compile(refusal_pattern, re.I)
    scored: list[dict] = []
    skipped = 0
    with requests.Session() as session:
        for row in rows:
            if row.get("skip"):
                skipped += 1
                continue
            t0 = time.perf_counter()
            resp = query_api(
                session, base_url, row, top_k, api_key, timeout, retries,
                retrieval_mode=retrieval_mode, alpha=alpha,
            )
            sc = score_row(row, resp, refusal_re, tol)
            sc["client_latency_ms"] = round((time.perf_counter() - t0) * 1000)
            scored.append(sc)
            if sleep:
                time.sleep(sleep)
    return scored, skipped


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{i}: invalid JSON ({e})")
            for need in ("id", "question"):
                if need not in row:
                    sys.exit(f"{path}:{i}: missing '{need}'")
            rows.append(row)
    return rows


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) and v <= 1 else str(v)


def print_report(overall: dict, by_cat: dict[str, dict], failures: list[dict]) -> None:
    cols = [
        "n",
        "hit_rate",
        "mrr",
        "answer_ok_rate",
        "refusal_accuracy",
        "false_refusal_rate",
        "citation_rate",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    short = {c: c.replace("_rate", "").replace("_accuracy", "_acc") for c in cols}
    print(f"\n{'category':<18}" + "".join(f"{short[c]:>15}" for c in cols))
    for name, agg in [("OVERALL", overall), *sorted(by_cat.items())]:
        print(f"{name:<18}" + "".join(f"{_fmt(agg.get(c)):>15}" for c in cols))
    if failures:
        print(f"\n{len(failures)} failing row(s):")
        for r in failures:
            why = r["error"] or (
                "not refused"
                if r["should_refuse"]
                else "retrieval miss"
                if r.get("hit") is False
                else "answer mismatch / refused"
            )
            print(f"  {r['id']:<22}{why:<28}top: {r.get('top_sources', [])[:3]}")


def run_ab_sweep(
    base_url: str,
    rows: list[dict],
    alphas: list[float],
    *,
    top_k: int = 5,
    api_key: Optional[str] = None,
    tol: int = 0,
    sleep: float = 0.0,
    refusal_pattern: str = DEFAULT_REFUSAL,
    timeout: float = 60,
    retries: int = 3,
) -> list[dict]:
    """
    Run the full eval for each alpha in `alphas` and return per-alpha aggregates.

    Returns a list of records::

        [{"alpha": 0.3, "overall": {...agg...}, "latency_p50_ms": ...}, ...]

    Save the result to data/test_sets/ab_sweep_<timestamp>.json.
    """
    sweep_results: list[dict] = []
    for alpha in alphas:
        print(f"\n── alpha={alpha} ──────────────────────────────")
        scored, skipped = run_eval(
            base_url, rows,
            top_k=top_k, api_key=api_key, tol=tol,
            sleep=sleep, refusal_pattern=refusal_pattern,
            timeout=timeout, retries=retries,
            retrieval_mode="hybrid", alpha=alpha,
        )
        agg = aggregate(scored)
        sweep_results.append({"alpha": alpha, "overall": agg, "rows_skipped": skipped})
        # Mini summary line
        print(
            f"  hit={_fmt(agg.get('hit_rate'))}  "
            f"mrr={_fmt(agg.get('mrr'))}  "
            f"answer_ok={_fmt(agg.get('answer_ok_rate'))}  "
            f"p50={agg.get('latency_p50_ms')}ms"
        )
    return sweep_results


def print_ab_table(sweep_results: list[dict]) -> None:
    """Print a compact comparison table for the A/B sweep."""
    cols = ["hit_rate", "mrr", "answer_ok_rate", "refusal_accuracy", "latency_p50_ms"]
    header = f"{'alpha':>8}" + "".join(f"{c:>18}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for rec in sweep_results:
        agg = rec["overall"]
        row = f"{rec['alpha']:>8.2f}" + "".join(f"{_fmt(agg.get(c)):>18}" for c in cols)
        print(row)


def compare(path_a: str, path_b: str, max_drop: float) -> int:
    with open(path_a, encoding="utf-8") as fa, open(path_b, encoding="utf-8") as fb:
        a, b = json.load(fa)["overall"], json.load(fb)["overall"]
    regressed = False
    print(f"{'metric':<20}{'before':>9}{'after':>9}{'delta':>9}")
    for k in HIGHER_IS_BETTER:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            continue
        d = vb - va
        bad = d < -max_drop
        regressed = regressed or bad
        print(f"{k:<20}{va:>9.3f}{vb:>9.3f}{d:>+9.3f}{'  <-- REGRESSION' if bad else ''}")
    return 1 if regressed else 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--eval-set", default="data/test_sets/eval_set_v2.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--api-key", default=os.getenv("EVAL_API_KEY"))
    ap.add_argument("--page-tolerance", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--refusal-regex", default=DEFAULT_REFUSAL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-hit", type=float, default=None)
    ap.add_argument("--min-answer-ok", type=float, default=None)
    ap.add_argument("--min-refusal", type=float, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--max-drop", type=float, default=0.03)
    ap.add_argument(
        "--ab-sweep", action="store_true",
        help="Run eval at multiple alpha values and print a comparison table.",
    )
    ap.add_argument(
        "--alphas", default="0.3,0.5,0.7,0.9",
        help="Comma-separated alpha values for --ab-sweep (default: 0.3,0.5,0.7,0.9).",
    )
    args = ap.parse_args(argv)

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.max_drop)

    if args.ab_sweep:
        alphas = [float(a.strip()) for a in args.alphas.split(",")]
        rows = load_rows(args.eval_set)
        sweep = run_ab_sweep(
            args.base_url, rows, alphas,
            top_k=args.top_k, api_key=args.api_key,
            tol=args.page_tolerance, sleep=args.sleep,
            refusal_pattern=args.refusal_regex,
        )
        print_ab_table(sweep)
        out = args.out or f"data/test_sets/ab_sweep_{datetime.now():%Y%m%d_%H%M}.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "git_sha": _git_sha(),
                    "base_url": args.base_url,
                    "eval_set": args.eval_set,
                    "alphas": alphas,
                    "sweep": sweep,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nA/B sweep report -> {out}")
        return 0

    rows = load_rows(args.eval_set)
    scored, skipped = run_eval(
        args.base_url,
        rows,
        top_k=args.top_k,
        api_key=args.api_key,
        tol=args.page_tolerance,
        sleep=args.sleep,
        refusal_pattern=args.refusal_regex,
    )
    overall = aggregate(scored)
    cats = defaultdict(list)
    for r in scored:
        cats[r["category"]].append(r)
    by_cat = {c: aggregate(v) for c, v in cats.items()}
    failures = [
        r
        for r in scored
        if r["error"]
        or (r["should_refuse"] and not r.get("refusal_correct"))
        or (not r["should_refuse"] and (r.get("hit") is False or not r.get("answer_ok")))
    ]
    print_report(overall, by_cat, failures)
    if skipped:
        print(f"\nskipped {skipped} template row(s) marked skip=true")

    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "base_url": args.base_url,
        "top_k": args.top_k,
        "eval_set": args.eval_set,
        "eval_set_sha256": hashlib.sha256(Path(args.eval_set).read_bytes()).hexdigest(),
        "rows_scored": len(scored),
        "rows_skipped": skipped,
    }
    try:
        meta["api_stats"] = requests.get(f"{args.base_url}/stats", timeout=5).json()
    except Exception:
        pass
    out = args.out or f"data/test_sets/api_eval_{datetime.now():%Y%m%d_%H%M}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        json.dumps(
            {"meta": meta, "overall": overall, "by_category": by_cat, "rows": scored},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nreport -> {out}")

    gates = [
        ("hit_rate", args.min_hit),
        ("answer_ok_rate", args.min_answer_ok),
        ("refusal_accuracy", args.min_refusal),
    ]
    failed = [
        f"{k} {overall[k]} < {floor}"
        for k, floor in gates
        if floor is not None and (overall[k] is None or overall[k] < floor)
    ]
    for msg in failed:
        print(f"GATE FAILED: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
