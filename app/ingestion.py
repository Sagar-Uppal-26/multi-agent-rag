import os
import pickle
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from rank_bm25 import BM25Okapi

VECTOR_STORE_DIR = "vector_store"
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")
FAISS_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")

# Local embedding model (runs on CPU, free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_pdf(file_path: str) -> int:
    """Loads a PDF, splits into chunks with metadata, and updates FAISS + BM25 stores."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    new_chunks = splitter.split_documents(raw_docs)

    filename = os.path.basename(file_path)
    for chunk in new_chunks:
        chunk.metadata["source"] = filename
        # Prepend filename so every chunk matches document-level queries
        chunk.page_content = f"Document: {filename}\n{chunk.page_content}"

    # Load existing chunks and remove old copies of this same file
    all_chunks: List[Document] = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            all_chunks = pickle.load(f)

    all_chunks = [c for c in all_chunks if c.metadata.get("source") != filename]
    all_chunks.extend(new_chunks)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    vector_store = FAISS.from_documents(all_chunks, embeddings)
    vector_store.save_local(FAISS_PATH)

    return len(new_chunks)


def hybrid_search(query: str, top_k: int = 4, source_filter: Optional[str] = None) -> List[Document]:
    """Combines FAISS dense retrieval and BM25 sparse retrieval using Reciprocal Rank Fusion."""
    if not os.path.exists(FAISS_PATH) or not os.path.exists(CHUNKS_PATH):
        return []

    with open(CHUNKS_PATH, "rb") as f:
        all_chunks: List[Document] = pickle.load(f)

    if source_filter:
        all_chunks = [c for c in all_chunks if c.metadata.get("source") == source_filter]
        if not all_chunks:
            return []

    # 1. Dense Search (FAISS)
    vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    search_kwargs = {"k": top_k * 2}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}
    dense_docs = vector_store.similarity_search(query, **search_kwargs)

    # 2. Sparse Search (BM25)
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: top_k * 2]
    sparse_docs = [all_chunks[i] for i in top_bm25_idx]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)

    ranked_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    return [doc_map[k] for k in ranked_keys]