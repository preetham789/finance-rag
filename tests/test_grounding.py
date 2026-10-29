from src.grounding import REFUSAL_TEXT, enforce, numbers_in, verify

REPO_CTX = (
    "The Monetary Policy Committee reduced the policy repo rate by 25 basis points "
    "to 6.25 per cent in February 2025."
)


def test_supported_answer_is_ok():
    ans = "The repo rate was cut by 25 bps to 6.25% in February 2025 [Source: RBI, Page 112]."
    rep = verify(ans, [REPO_CTX], [("RBI", 112)])
    assert rep.ok and not rep.unsupported_numbers and not rep.invalid_citations


def test_unsupported_number_is_flagged():
    ans = "The repo rate was cut to 6.5% [Source: RBI, Page 112]."
    rep = verify(ans, [REPO_CTX], [("RBI", 112)])
    assert rep.unsupported_numbers == ["6.5"]
    assert not rep.ok


def test_thousands_and_indian_grouping_match_plain_digits():
    assert numbers_in("607,979 staff") == {"607979"}
    assert numbers_in("Rs 1,23,456 crore") == {"123456"}
    rep = verify(
        "TCS has 607,979 employees [Source: TCS, Page 24].",
        ["Total headcount stood at 607979."],
        [("TCS", 24)],
    )
    assert rep.ok


def test_decimal_equivalence():
    rep = verify(
        "CRR is 4% [Source: RBI, Page 70].", ["CRR was set at 4.0 per cent."], [("RBI", 70)]
    )
    assert rep.ok


def test_year_range_in_context_supports_fiscal_year_in_answer():
    rep = verify(
        "FY2025 growth was 6.5% [Source: RBI, Page 3].",
        ["Growth in 2024-25 was 6.5 per cent."],
        [("RBI", 3)],
    )
    assert rep.ok


def test_invalid_citation_is_flagged():
    rep = verify(
        "Headcount is 317,240 [Source: TCS, Page 99].",
        ["Infosys had 317,240 employees."],
        [("Infosys", 99)],
    )
    assert rep.invalid_citations == [("TCS", 99)]
    assert not rep.ok


def test_company_match_is_loose():
    rep = verify(
        "NPA fell to 1.2% [Source: ICICI Bank, Page 8].",
        ["Gross NPA ratio was 1.2 per cent."],
        [("ICICI", 8)],
    )
    assert not rep.invalid_citations


def test_uncited_answer_flagged_only_when_required():
    ans = "The repo rate is 6.25%."
    assert verify(ans, [REPO_CTX]).uncited
    assert not verify(ans, [REPO_CTX], require_citation=False).uncited


def test_refusal_skips_all_checks():
    rep = verify("The provided documents do not contain sufficient information.", [])
    assert rep.refused and rep.ok and rep.checked_numbers == 0


def test_citations_passages_and_list_markers_are_not_claims():
    ans = "1. According to Passage 3, rate is 6.25% [Source: RBI, Page 112].\n2. Done."
    rep = verify(ans, [REPO_CTX], [("RBI", 112)])
    assert rep.unsupported_numbers == []


def test_enforce_modes():
    rep = verify("Rate is 9% [Source: RBI, Page 1].", [REPO_CTX], [("RBI", 1)])
    assert not rep.ok
    assert enforce("Rate is 9%.", rep) == "Rate is 9%."
    assert enforce("Rate is 9%.", rep, strict=True) == REFUSAL_TEXT
