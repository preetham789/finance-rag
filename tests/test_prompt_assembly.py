from unittest.mock import MagicMock, patch

from src.generation.chain import (
    FinanceRAGChain,
    build_sources,
    build_user_message,
    format_context,
)


def test_prompt_assembly_structure():
    chunks = [
        {"company": "RBI", "page_number": 12, "text": "Repo rate is 6.50%."},
        {"company": "TCS", "page_number": 5, "text": "TCS headcount is 600,000."},
    ]
    formatted = format_context(chunks)
    assert "[Passage 1 | RBI | Page 12]" in formatted
    assert "[Passage 2 | TCS | Page 5]" in formatted

    user_msg = build_user_message("What is repo rate?", formatted)
    assert "<context_passages>" in user_msg
    assert "</context_passages>" in user_msg
    assert "Question: What is repo rate?" in user_msg


def test_build_sources():
    chunks = [
        {
            "company": "RBI",
            "page_number": 12,
            "source_file": "rbi.pdf",
            "hybrid_score": 0.91,
            "text": "Detailed text " * 20,
        }
    ]
    sources = build_sources(chunks)
    assert len(sources) == 1
    assert sources[0]["company"] == "RBI"
    assert sources[0]["score"] == 0.91
    assert sources[0]["preview"].endswith("...")


def test_rag_chain_with_fake_llm():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "company": "RBI",
            "page_number": 12,
            "source_file": "rbi.pdf",
            "score": 0.9,
            "text": "The repo rate was kept at 6.50% by the Monetary Policy Committee.",
        }
    ]

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The repo rate is 6.50% [Source: RBI, Page 12]."
    mock_resp.choices = [mock_choice]
    mock_resp.usage.total_tokens = 50

    with patch("src.generation.chain.make_llm_client") as mock_make:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_make.return_value = (mock_client, "fake-model")

        chain = FinanceRAGChain(retriever=mock_retriever, api_key="dummy")
        res = chain.query("What is the repo rate?")

        assert "6.50%" in res["answer"]
        assert len(res["sources"]) == 1
        assert res["grounding"]["ok"] is True
