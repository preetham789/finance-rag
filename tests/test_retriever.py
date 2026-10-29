import pytest
from unittest.mock import MagicMock

from src.retrieval.retriever import (
    FinanceRetriever,
    deduplicate_chunks,
    matches_where_filter,
    tokenize_financial_text,
    understand_query,
)


class DummyModel:
    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        return np.zeros((len(texts), 384))


@pytest.fixture
def dummy_retriever():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["RBI kept repo rate at 6.5%.", "HDFC Net Interest Margin expanded."]],
        "metadatas": [[
            {"company": "RBI", "page_number": 10, "source_file": "rbi.pdf", "doc_type": "rbi"},
            {"company": "HDFC", "page_number": 20, "source_file": "hdfc.pdf", "doc_type": "annual_report"},
        ]],
        "distances": [[0.2, 0.4]],
    }
    model = DummyModel()
    return FinanceRetriever(collection=mock_collection, model=model, top_k=2)


def test_retriever_initialization(dummy_retriever):
    assert dummy_retriever.top_k == 2
    assert dummy_retriever.reranker is None
    assert dummy_retriever.rerank_factor == 4


def test_dense_search(dummy_retriever):
    res = dummy_retriever.dense_search("repo rate")
    assert len(res) == 2
    assert res[0]["company"] == "RBI"
    assert res[0]["score"] == 0.8  # 1 - 0.2


def test_hybrid_search(dummy_retriever):
    res = dummy_retriever.hybrid_search("repo rate", alpha=0.7)
    assert len(res) == 2
    assert "hybrid_score" in res[0]
    assert "bm25_score" in res[0]


def test_rerank_fallback_when_none(dummy_retriever):
    # Without reranker loaded, should log warning and fall back to hybrid search
    res = dummy_retriever.rerank_search("repo rate", alpha=0.7)
    assert len(res) == 2
    assert "hybrid_score" in res[0]


def test_rerank_with_mock_reranker(dummy_retriever):
    mock_reranker = MagicMock()
    # Predict returns high score for 2nd doc, low score for 1st doc
    mock_reranker.predict.return_value = [0.1, 0.9]
    dummy_retriever.reranker = mock_reranker

    res = dummy_retriever.rerank_search("repo rate", alpha=0.7)
    assert len(res) == 2
    assert res[0]["company"] == "HDFC"  # Ranked first due to 0.9 score
    assert res[0]["rerank_score"] == 0.9


def test_retrieve_sweep_no_mutation(dummy_retriever):
    alphas = [0.1, 0.9]
    sweep = dummy_retriever.retrieve_sweep("repo rate", alphas=alphas)
    assert len(sweep) == 2
    assert sweep[0]["alpha"] == 0.1
    assert sweep[1]["alpha"] == 0.9
    # Both sweeps return chunks with hybrid_score present
    assert "hybrid_score" in sweep[0]["chunks"][0]
    assert "hybrid_score" in sweep[1]["chunks"][0]
    # The sweep dict is distinct per alpha (no shared mutation)
    assert sweep[0] is not sweep[1]


def test_retrieve_mode_selection(dummy_retriever):
    # Use enable_query_understanding=False so no company filter is injected,
    # ensuring both mock docs are returned.
    r_dense = dummy_retriever.retrieve("repo rate", mode="dense", enable_query_understanding=False)
    assert len(r_dense) == 2

    r_hybrid = dummy_retriever.retrieve("repo rate", mode="hybrid", alpha=0.5, enable_query_understanding=False)
    assert "hybrid_score" in r_hybrid[0]

    r_rerank = dummy_retriever.retrieve("repo rate", mode="rerank", enable_query_understanding=False)
    assert len(r_rerank) == 2


def test_tokenize_financial_text():
    tokens = tokenize_financial_text("RBI kept repo rate at 6.25%, GNPA reduced to 2.8% in FY24.")
    assert "6.25%" in tokens
    assert "2.8%" in tokens
    assert "gnpa" in tokens
    assert "fy24" in tokens


def test_understand_query():
    info = understand_query("What was RBI's repo rate in 2024?")
    assert info["company"] == "RBI"
    assert info["doc_type"] == "rbi"
    assert info["fiscal_year"] == "2024"

    info_tcs = understand_query("What was TCS revenue in FY24?")
    assert info_tcs["company"] == "TCS"
    assert info_tcs["doc_type"] == "company"


def test_matches_where_filter():
    chunk = {"company": "HDFC Bank", "doc_type": "company"}
    assert matches_where_filter(chunk, {"company": "HDFC Bank"})
    assert not matches_where_filter(chunk, {"company": "SBI"})
    assert matches_where_filter(chunk, {"$and": [{"company": "HDFC Bank"}, {"doc_type": "company"}]})
    assert not matches_where_filter(chunk, {"$and": [{"company": "HDFC Bank"}, {"doc_type": "rbi"}]})


def test_deduplicate_chunks():
    chunks = [
        {"text": "The repo rate was decided by the monetary policy committee to remain at 6.50 percent."},
        {"text": "The repo rate was decided by the monetary policy committee to remain at 6.50 percent."},  # Exact dup
        {"text": "Completely different chunk text with unique data on banking sector NPAs."},
    ]
    deduped = deduplicate_chunks(chunks, overlap_threshold=0.85)
    assert len(deduped) == 2
