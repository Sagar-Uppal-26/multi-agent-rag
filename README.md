# Multi-Agent RAG Document Assistant

A document question-answering system built with LangGraph, hybrid retrieval (FAISS + BM25), FastAPI, and Streamlit. Rather than passing retrieved chunks directly to an LLM, the pipeline routes incoming queries, combines semantic and keyword search, and verifies answers against the source text before returning them.

**Live Demo:** [Open Streamlit App](https://multi-agent-rag-blm3b93cqz2saibtqpcefo.streamlit.app/)

## How It Works

When a user submits a query, it passes through a stateful LangGraph workflow:

1. **Query Router:** Classifies whether the prompt requires document retrieval or can be answered directly as general conversation.
2. **Hybrid Retriever:** Queries both a FAISS vector index (dense embeddings via `all-MiniLM-L6-v2`) and a BM25 index (exact keyword matching), merging the top results with optional filename filtering.
3. **Grounded Generator:** Generates an answer strictly from the retrieved context and cites the source document and page number.
4. **Hallucination Grader:** Verifies that every claim in the generated response is supported by the retrieved chunks. If unsupported claims are detected, it triggers a regeneration loop (capped at 2 retries).

## Project Structure

```text
multi-agent-rag/
├── app/
│   ├── ingestion.py      # PDF chunking, metadata tagging, FAISS & BM25 indexing
│   ├── graph.py          # LangGraph state machine and agent nodes
│   └── main.py           # FastAPI REST endpoints (/ingest, /query, /health)
├── eval/
│   └── evaluate.py       # Automated faithfulness and answer relevance evaluation
├── frontend.py           # Streamlit web interface with live agent trace metrics
├── Dockerfile            # Container setup for the FastAPI service
└── requirements.txt
