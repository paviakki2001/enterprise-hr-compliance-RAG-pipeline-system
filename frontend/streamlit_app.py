from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from api_client import ask_question, health_check, APIClientError
from config import settings

st.set_page_config(page_title=settings.APP_TITLE, page_icon="🤖", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title(settings.APP_TITLE)
st.caption("Week 4 user interface regenerated from your Week 3 Semantic RAG + FastAPI notebook")

with st.sidebar:
    st.subheader("Backend Connection")
    status = health_check()
    if status["ok"]:
        st.success("Backend is reachable")
        data = status.get("data", {})
        if data:
            st.write(f"**Embedding model:** {data.get('embedding_model', 'N/A')}")
            st.write(f"**LLM model:** {data.get('llm_model', 'N/A')}")
            st.write(f"**Reranker enabled:** {data.get('reranker_enabled', 'N/A')}")
            st.write(f"**Vectors available:** {data.get('vectors_available', 'N/A')}")
            st.write(f"**Vectorstore mode:** {data.get('vectorstore_mode', 'N/A')}")
    else:
        st.error("Backend is not reachable")
        st.caption(status.get("detail", ""))

    st.divider()
    top_k_retrieval = st.slider("Top K Retrieval", 1, 20, settings.DEFAULT_TOP_K_RETRIEVAL)
    top_k_context = st.slider("Top K Context", 1, 10, settings.DEFAULT_TOP_K_CONTEXT)
    show_citations = st.toggle("Show citations", value=settings.SHOW_CITATIONS_DEFAULT)

    st.divider()
    st.write("Optional metadata filters")
    department = st.text_input("department")
    document_type = st.text_input("document_type")
    category = st.text_input("category")
    region = st.text_input("region")
    year = st.text_input("year")
    source = st.text_input("source")
    file_name = st.text_input("file_name")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

st.markdown("### Ask a question")
question = st.text_area(
    "Enter your question",
    placeholder="Example: What do the retrieved HR or compliance documents say about onboarding policy requirements?",
    height=130,
)

if st.button("Submit", use_container_width=True):
    filters = {
        k: v for k, v in {
            "department": department,
            "document_type": document_type,
            "category": category,
            "region": region,
            "year": year,
            "source": source,
            "file_name": file_name,
        }.items() if str(v).strip()
    }
    filters = filters if filters else None

    try:
        result = ask_question(
            question=question,
            top_k_retrieval=top_k_retrieval,
            top_k_context=top_k_context,
            filters=filters,
        )
        st.session_state.history.append(result)
    except APIClientError as exc:
        st.error(str(exc))

if st.session_state.history:
    latest = st.session_state.history[-1]
    st.markdown("### Latest Answer")
    st.success(latest["answer"])
    col1, col2 = st.columns(2)
    col1.metric("Latency (seconds)", latest["latency_seconds"])
    col2.metric("Retrieved chunks", latest["retrieved_count"])

    if show_citations and latest["citations"]:
        st.markdown("### Citations")
        for i, citation in enumerate(latest["citations"], start=1):
            with st.expander(f"Citation {i} | Source: {citation.get('source', 'N/A')} | Chunk: {citation.get('chunk_id', 'N/A')}"):
                st.json(citation)

    st.markdown("### Query History")
    history_df = pd.DataFrame([
        {
            "question": row.get("question", ""),
            "latency_seconds": row.get("latency_seconds", None),
            "retrieved_count": row.get("retrieved_count", None),
        }
        for row in st.session_state.history
    ])
    st.dataframe(history_df, use_container_width=True)

    if not history_df.empty and "latency_seconds" in history_df.columns:
        st.bar_chart(history_df["latency_seconds"])
else:
    st.info("No queries submitted yet.")
