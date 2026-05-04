"""
RAG chain that supports OpenAI-compatible providers such as OpenAI, Groq,
and local Ollama.
"""

import logging
from typing import Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    DefaultHttpxClient,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial analyst assistant with access to \
Indian company annual reports and RBI publications.

STRICT RULES:
1. Answer ONLY using the provided context passages. Never use outside knowledge.
2. If the context does not contain enough information, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
3. Always cite your sources using [Source: <company>, Page <N>] format inline.
4. For numerical data (revenue, ratios, percentages), quote the exact figure from context.
5. Keep answers concise and factual. No speculation."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a prompt-friendly context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        company = chunk.get("company", "Unknown")
        page = chunk.get("page_number", "?")
        lines.append(f"[Passage {i} | {company} | Page {page}]")
        lines.append(chunk["text"].strip())
        lines.append("")
    return "\n".join(lines)


def build_user_message(question: str, context: str) -> str:
    """Build the user turn for the LLM call."""
    return f"""Context passages from financial documents:

{context}

Question: {question}

Answer based strictly on the context above. Cite sources inline."""


def build_sources(chunks: list[dict]) -> list[dict]:
    """Extract source metadata for display in the CLI."""
    return [
        {
            "company": c["company"],
            "page_number": c["page_number"],
            "source_file": c["source_file"],
            "score": c.get("hybrid_score") or c.get("score", 0),
            "preview": c["text"][:150] + "...",
        }
        for c in chunks
    ]


def make_llm_client(api_key: str, provider: str = "groq") -> tuple[OpenAI, str]:
    """
    Return an OpenAI-compatible client plus model name for the chosen provider.

    Using an explicit DefaultHttpxClient keeps the SDK compatible with the
    currently installed httpx version.
    """
    http_client = DefaultHttpxClient()

    if provider == "groq":
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=http_client,
        )
        model = "llama-3.1-8b-instant"
    elif provider == "openai":
        client = OpenAI(
            api_key=api_key,
            http_client=http_client,
        )
        model = "gpt-4o-mini"
    elif provider == "ollama":
        client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            http_client=http_client,
        )
        model = "llama3.2"
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Use 'groq', 'openai', or 'ollama'."
        )

    return client, model


def format_llm_error_message(exc: Exception, provider: str) -> str:
    """Convert provider/API failures into short user-facing guidance."""
    provider_label = provider.capitalize()

    if isinstance(exc, AuthenticationError):
        return (
            f"{provider_label} rejected the API key. Check the matching key in "
            "your .env file and make sure it is valid."
        )

    if isinstance(exc, RateLimitError):
        error_body = getattr(exc, "body", {}) or {}
        if isinstance(error_body, dict):
            error_info = error_body.get("error", {}) or {}
            if error_info.get("code") == "insufficient_quota":
                return (
                    f"{provider_label} returned insufficient_quota (HTTP 429). "
                    "Add billing or credits to that account, or switch to a "
                    "funded API key, then rerun the query."
                )

        return (
            f"{provider_label} rate-limited the request (HTTP 429). Wait a "
            "moment and try again."
        )

    if isinstance(exc, APIConnectionError):
        return (
            f"Could not reach the {provider_label} API. Check your network, "
            "firewall, or proxy settings, then try again."
        )

    if isinstance(exc, APIStatusError):
        return (
            f"{provider_label} returned an API error (HTTP {exc.status_code}). "
            "Please try again in a moment."
        )

    return f"The {provider_label} request failed unexpectedly."


class LLMGenerationError(RuntimeError):
    """Raised when retrieval succeeds but answer generation fails."""

    def __init__(
        self,
        message: str,
        *,
        question: str,
        sources: list[dict],
        chunks: list[dict],
        retrieval_mode: str,
        model: str,
    ):
        super().__init__(message)
        self.question = question
        self.sources = sources
        self.chunks = chunks
        self.retrieval_mode = retrieval_mode
        self.model = model

    def as_result(self) -> dict:
        """Return a result-shaped payload for CLI rendering."""
        return {
            "answer": f"ERROR: {self}",
            "sources": self.sources,
            "chunks": self.chunks,
            "question": self.question,
            "retrieval_mode": self.retrieval_mode,
            "model": self.model,
            "tokens_used": "?",
        }


class FinanceRAGChain:
    """Retrieve chunks, call an LLM, and return an answer with sources."""

    def __init__(
        self,
        retriever,
        api_key: str,
        provider: str = "groq",
        temperature: float = 0.0,
    ):
        self.retriever = retriever
        self.temperature = temperature
        self.provider = provider
        self.llm, self.model = make_llm_client(api_key, provider)

    def query(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        where_filter: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        """Run the full RAG flow for one question."""
        chunks = self.retriever.retrieve(
            query=question,
            mode=retrieval_mode,
            where_filter=where_filter,
        )

        if not chunks:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": [],
                "chunks": [],
                "question": question,
                "retrieval_mode": retrieval_mode,
                "model": self.model,
                "tokens_used": "?",
            }

        if verbose:
            print(f"\nRetrieved {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get("hybrid_score") or chunk.get("score", 0)
                print(
                    f"  {i}. [{chunk['company']} p.{chunk['page_number']}] "
                    f"score={score:.3f}  {chunk['text'][:80]}..."
                )

        context = format_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context)},
        ]
        sources = build_sources(chunks)

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.exception("LLM generation failed for question: %s", question)
            raise LLMGenerationError(
                format_llm_error_message(exc, self.provider),
                question=question,
                sources=sources,
                chunks=chunks,
                retrieval_mode=retrieval_mode,
                model=self.model,
            ) from exc

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "question": question,
            "retrieval_mode": retrieval_mode,
            "model": self.model,
            "tokens_used": response.usage.total_tokens,
        }
