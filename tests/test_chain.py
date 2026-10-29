from unittest.mock import MagicMock, patch
from openai import RateLimitError
import httpx

from src.generation.chain import FinanceRAGChain


def test_chain_query_timings_and_format():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "company": "RBI",
            "page_number": 10,
            "source_file": "rbi.pdf",
            "score": 0.95,
            "text": "The repo rate was kept unchanged at 6.50% by the MPC.",
        }
    ]

    mock_choice = MagicMock()
    mock_choice.message.content = "The repo rate was unchanged at 6.50% [Source: RBI, Page 10]."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.total_tokens = 42

    with patch("src.generation.chain.make_llm_client") as mock_make_llm:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_make_llm.return_value = (mock_client, "mock-model")

        chain = FinanceRAGChain(retriever=mock_retriever, api_key="test-key", provider="groq")
        result = chain.query("What is the repo rate?")

        assert "answer" in result
        assert "timings" in result
        assert "retrieval_ms" in result["timings"]
        assert "llm_ms" in result["timings"]
        assert "grounding_ms" in result["timings"]
        assert result["tokens_used"] == 42
        assert result["grounding"] is not None


def test_chain_retry_on_rate_limit():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "company": "RBI",
            "page_number": 1,
            "source_file": "rbi.pdf",
            "score": 0.9,
            "text": "Some text",
        }
    ]

    # First attempt raises RateLimitError, second succeeds
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp_429 = httpx.Response(429, request=req, headers={"Retry-After": "0.01"})
    rate_err = RateLimitError("Rate limit exceeded", response=resp_429, body={"error": {}})

    mock_choice = MagicMock()
    mock_choice.message.content = "Answer [Source: RBI, Page 1]."
    success_resp = MagicMock()
    success_resp.choices = [mock_choice]
    success_resp.usage.total_tokens = 25

    with patch("src.generation.chain.make_llm_client") as mock_make_llm:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [rate_err, success_resp]
        mock_make_llm.return_value = (mock_client, "mock-model")

        chain = FinanceRAGChain(retriever=mock_retriever, api_key="test-key", max_retries=2)
        result = chain.query("test question")

        assert mock_client.chat.completions.create.call_count == 2
        assert "Answer" in result["answer"]
