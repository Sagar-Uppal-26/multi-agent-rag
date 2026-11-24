import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingestion import ingest_pdf
from app.graph import rag_graph

app = FastAPI(title="Multi-Agent RAG Document Intelligence API", version="1.0.0")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    num_chunks = ingest_pdf(file_path)
    return {"filename": file.filename, "chunks_indexed": num_chunks, "status": "indexed"}


@app.post("/query")
async def query_documents(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "source_filter": request.source_filter,
        "route": "",
        "context": [],
        "answer": "",
        "is_grounded": False,
        "retries": 0,
    }
    result = await rag_graph.ainvoke(initial_state)
    return {
        "query": result["query"],
        "route": result["route"],
        "answer": result["answer"],
        "is_grounded": result["is_grounded"],
        "retries": result["retries"],
        "sources_used": len(result["context"]),
    }