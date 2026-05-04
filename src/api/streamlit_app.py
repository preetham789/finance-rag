# src/api/streamlit_app.py
"""
Streamlit chat UI for the Finance RAG system.

Talks to the FastAPI backend at localhost:8000.
Recruiters open this in their browser and ask questions directly.
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page config ──
st.set_page_config(
    page_title="Finance RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Finance RAG System")
st.caption("Q&A over Indian company annual reports and RBI publications")

# ── Sidebar ──
with st.sidebar:
    st.header("Search filters")

    filter_type = st.radio(
        "Filter by",
        ["None", "Company", "Document type"],
        index=0,
    )

    company_filter  = None
    doc_type_filter = None

    if filter_type == "Company":
        company_filter = st.selectbox("Company", [
            "TCS", "Infosys", "Wipro", "HCL Technologies",
            "Reliance", "HDFC Bank", "ICICI Bank", "Axis Bank",
            "SBI", "Bajaj Finance", "Asian Paints", "L&T",
            "Maruti Suzuki", "Sun Pharma", "ITC",
        ])
    elif filter_type == "Document type":
        doc_type_filter = st.selectbox("Type", ["rbi", "company"])

    top_k = st.slider("Sources to retrieve", 3, 10, 5)

    st.divider()
    st.subheader("System status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API online")
        st.metric("Vectors", f"{health['vectors']:,}")
    except Exception:
        st.error("API offline — start with: uvicorn src.api.main:app")

    st.divider()
    st.subheader("Example questions")
    examples = [
        "What was RBI's repo rate in February 2025?",
        "What is TCS's primary business?",
        "What are Reliance's main business segments?",
        "What was India's inflation in 2024-25?",
        "How many employees does Infosys have?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

# ── Chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"Sources ({len(msg['sources'])} retrieved)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['company']}** — Page {src['page_number']} "
                        f"— relevance {src['score']:.3f}"
                    )
                    st.caption(src["preview"])

# ── Chat input ──
prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask about Indian companies or RBI policy...") or prefill

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching 33,000+ chunks..."):
            try:
                payload = {
                    "question": question,
                    "top_k":    top_k,
                }
                if company_filter:
                    payload["company"] = company_filter
                if doc_type_filter:
                    payload["doc_type"] = doc_type_filter

                resp = requests.post(
                    f"{API_URL}/query",
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                answer  = data["answer"]
                sources = data["sources"]
                meta    = (f"Model: `{data['model']}` | "
                           f"Latency: {data['latency_ms']}ms | "
                           f"Tokens: {data['tokens_used']}")

                st.markdown(answer)
                st.caption(meta)

                with st.expander(f"Sources ({len(sources)} retrieved)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**[{i}] {src['company']}** — Page {src['page_number']} "
                            f"— relevance {src['score']:.3f}"
                        )
                        st.caption(src["preview"])

                # Save to history
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Run: uvicorn src.api.main:app --reload")
            except Exception as e:
                st.error(f"Error: {e}")