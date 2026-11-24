import os
import streamlit as st

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from app.ingestion import ingest_pdf
from app.graph import rag_graph

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("Multi-Agent RAG Document Assistant")
st.caption("LangGraph workflow with FAISS + BM25 hybrid retrieval and hallucination grading")

with st.sidebar:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if st.button("Index PDF", use_container_width=True, disabled=uploaded_file is None):
        with st.spinner("Indexing document..."):
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            chunks = ingest_pdf(file_path)
            st.success(f"Indexed {uploaded_file.name} ({chunks} chunks)")

    st.divider()
    source_filter = st.text_input("Filter by filename (optional)", placeholder="example.pdf").strip() or None

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            m = msg["meta"]
            with st.expander("Agent Trace"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Route", m["route"].upper())
                c2.metric("Grounded", "Yes" if m["is_grounded"] else "No")
                c3.metric("Retries", m["retries"])
                c4.metric("Chunks", m["sources_used"])

if query := st.chat_input("Ask a question about your uploaded PDFs..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            result = rag_graph.invoke({
                "query": query,
                "source_filter": source_filter,
                "route": "",
                "context": [],
                "answer": "",
                "is_grounded": False,
                "retries": 0,
            })

            answer = result["answer"]
            meta = {
                "route": result["route"],
                "is_grounded": result["is_grounded"],
                "retries": result["retries"],
                "sources_used": len(result["context"]),
            }

            st.markdown(answer)
            with st.expander("Agent Trace"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Route", meta["route"].upper())
                c2.metric("Grounded", "Yes" if meta["is_grounded"] else "No")
                c3.metric("Retries", meta["retries"])
                c4.metric("Chunks", meta["sources_used"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": meta,
            })