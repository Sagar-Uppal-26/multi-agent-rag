import os
from typing import List, Optional, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from app.ingestion import hybrid_search

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1,
    max_retries=1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def get_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


class AgentState(TypedDict):
    query: str
    source_filter: Optional[str]
    route: str
    context: List[str]
    answer: str
    is_grounded: bool
    retries: int


def router_node(state: AgentState) -> AgentState:
    prompt = (
        "Classify the user query into one of two routes: 'retrieve' (needs information from uploaded documents) "
        "or 'direct' (general greeting or chitchat). Reply with ONLY one word: retrieve or direct.\n\n"
        f"Query: {state['query']}"
    )
    decision = get_text(llm.invoke(prompt).content).strip().lower()
    route = "direct" if "direct" in decision else "retrieve"
    return {**state, "route": route, "retries": 0}


def retriever_node(state: AgentState) -> AgentState:
    docs = hybrid_search(state["query"], top_k=6, source_filter=state.get("source_filter"))
    context = [
        f"[Source: {d.metadata.get('source', 'unknown')} | Page: {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    ]
    return {**state, "context": context}


def direct_node(state: AgentState) -> AgentState:
    response = get_text(llm.invoke(state["query"]).content)
    return {**state, "answer": response, "is_grounded": True, "context": []}


def generator_node(state: AgentState) -> AgentState:
    if not state["context"]:
        return {**state, "answer": "No relevant documents found. Please upload a PDF first.", "is_grounded": True}

    context_str = "\n\n---\n\n".join(state["context"])
    prompt = (
        "You are a precise document intelligence assistant. Answer the question strictly using ONLY the provided context. "
        "If the context does not contain the answer, say 'I cannot find this information in the provided documents.' "
        "Cite the source and page number where applicable.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {state['query']}"
    )
    answer = get_text(llm.invoke(prompt).content)
    return {**state, "answer": answer, "retries": state.get("retries", 0) + 1}


def grader_node(state: AgentState) -> AgentState:
    answer_text = get_text(state["answer"])
    if not state["context"] or "cannot find this information" in answer_text.lower():
        return {**state, "is_grounded": True}

    context_str = "\n\n".join(state["context"])
    prompt = (
        "You are a strict hallucination grader. Check if the Answer is fully supported by the Context.\n"
        "Reply with ONLY 'YES' if grounded, or 'NO' if it contains unsupported claims.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Answer:\n{answer_text}"
    )
    verdict = get_text(llm.invoke(prompt).content).strip().upper()
    is_grounded = "YES" in verdict
    return {**state, "is_grounded": is_grounded}


def route_edge(state: AgentState) -> str:
    return "direct" if state["route"] == "direct" else "retriever"


def grade_edge(state: AgentState) -> str:
    if state["is_grounded"] or state["retries"] >= 2:
        return END
    return "generator"


builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("retriever", retriever_node)
builder.add_node("direct", direct_node)
builder.add_node("generator", generator_node)
builder.add_node("grader", grader_node)

builder.set_entry_point("router")
builder.add_conditional_edges("router", route_edge, {"direct": "direct", "retriever": "retriever"})
builder.add_edge("direct", END)
builder.add_edge("retriever", "generator")
builder.add_edge("generator", "grader")
builder.add_conditional_edges("grader", grade_edge, {"generator": "generator", END: END})

rag_graph = builder.compile()