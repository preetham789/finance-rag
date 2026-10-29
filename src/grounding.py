"""Deterministic post-generation checks for a RAG answer.

Two failure modes that a regex catches faster and more reliably than an LLM judge:
  1. a number in the answer that appears nowhere in the retrieved context
  2. a [Source: <company>, Page N] tag pointing at a chunk that was never retrieved

Pure functions: no I/O, no model calls, cheap enough to run on every request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional, Sequence

CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+?)\s*,\s*Page\s*(\d+)\s*\]", re.I)
REFUSAL_RE = re.compile(
    r"do(?:es)? not contain sufficient information|insufficient information", re.I
)
REFUSAL_TEXT = (
    "The provided documents do not contain sufficient information "
    "to answer this reliably."
)

_PASSAGE_RE = re.compile(r"\bPassage\s+\d+\b", re.I)
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.M)
# 607,979 (international) and 1,23,456 (Indian lakh/crore grouping)
_GROUPED_RE = re.compile(r"\b\d{1,3}(?:,\d{2,3})+(?!\d)")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s*[-\u2013/]\s*(\d{2,4})\b")


def _strip_grouping(text: str) -> str:
    return _GROUPED_RE.sub(lambda m: m.group(0).replace(",", ""), text)


def _canon(token: str) -> str:
    """'4.0' -> '4', '6.250' -> '6.25', '100' -> '100'."""
    try:
        return format(Decimal(token).normalize(), "f")
    except InvalidOperation:
        return token


def numbers_in(text: str, *, answer: bool = False) -> set[str]:
    """Canonical numeric tokens in `text`.

    answer=True strips citation tags, 'Passage N' mentions and list markers first,
    so they are not mistaken for claims. For context text, 2024-25 also yields 2025.
    """
    if answer:
        text = CITATION_RE.sub(" ", text)
        text = _PASSAGE_RE.sub(" ", text)
        text = _LIST_MARKER_RE.sub("", text)
    text = _strip_grouping(text)
    nums = {_canon(m) for m in _NUM_RE.findall(text)}
    if not answer:
        for m in _YEAR_RANGE_RE.finditer(text):
            start, end = m.group(1), m.group(2)
            if len(end) == 2:
                year = int(start[:2] + end)
                if year < int(start):
                    year += 100
                nums.add(str(year))
            else:
                nums.add(end)
    return nums


@dataclass
class GroundingReport:
    refused: bool = False
    checked_numbers: int = 0
    unsupported_numbers: list[str] = field(default_factory=list)
    invalid_citations: list[tuple[str, int]] = field(default_factory=list)
    uncited: bool = False

    @property
    def ok(self) -> bool:
        return self.refused or not (
            self.unsupported_numbers or self.invalid_citations or self.uncited
        )

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "refused": self.refused,
            "checked_numbers": self.checked_numbers,
            "unsupported_numbers": self.unsupported_numbers,
            "invalid_citations": self.invalid_citations,
            "uncited": self.uncited,
        }


def _norm_company(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _company_match(a: str, b: str) -> bool:
    x, y = _norm_company(a), _norm_company(b)
    return bool(x and y and (x in y or y in x))


def verify(
    answer: str,
    context_chunks: Iterable[str],
    retrieved: Optional[Sequence[tuple[str, int]]] = None,
    *,
    require_citation: bool = True,
) -> GroundingReport:
    """Check an answer against the chunks it was generated from.

    context_chunks: text of every chunk that went into the prompt.
    retrieved:      (company, page_number) for each of those chunks, if you want
                    cited [Source: X, Page N] tags validated as well.
    """
    report = GroundingReport()
    if REFUSAL_RE.search(answer):
        report.refused = True
        return report

    ctx_numbers: set[str] = set()
    for chunk in context_chunks:
        ctx_numbers |= numbers_in(chunk)
    ans_numbers = numbers_in(answer, answer=True)
    report.checked_numbers = len(ans_numbers)
    report.unsupported_numbers = sorted(
        ans_numbers - ctx_numbers, key=lambda s: (float(s), s)
    )

    cites = CITATION_RE.findall(answer)
    if require_citation and not cites:
        report.uncited = True
    if retrieved is not None:
        pool = list(retrieved)
        for company, page in cites:
            if not any(
                int(page) == int(p) and _company_match(company, c) for c, p in pool
            ):
                report.invalid_citations.append((company, int(page)))
    return report


def enforce(answer: str, report: GroundingReport, *, strict: bool = False) -> str:
    """strict=False: return the answer untouched (log the report instead).
    strict=True: replace the answer with a refusal on any violation."""
    return answer if (report.ok or not strict) else REFUSAL_TEXT
