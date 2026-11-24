import json
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.graph import rag_graph

load_dotenv()

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

TEST_QUESTIONS = [
    "What is the main topic of the uploaded document?",
    "Summarize the key findings or conclusions mentioned in the document."
]


def score_both(query: str, context: str, answer: str) -> dict:
    prompt = (
        "Evaluate the RAG output on two metrics from 0.0 to 1.0:\n"
        "1. retrieval_relevance: How relevant is the Context to the Question?\n"
        "2. faithfulness: Is the Answer fully supported by the Context without hallucinations?\n"
        'Reply ONLY with valid JSON like: {"retrieval_relevance": 0.9, "faithfulness": 1.0}\n\n'
        f"Question: {query}\nContext: {context}\nAnswer: {answer}"
    )
    raw = judge_llm.invoke(prompt).content.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"retrieval_relevance": 0.0, "faithfulness": 0.0}


def run_evaluation():
    results = []
    for idx, q in enumerate(TEST_QUESTIONS):
        if idx > 0:
            print("Waiting 65s to respect Gemini free-tier rate limit...")
            time.sleep(65)

        print(f"Evaluating question {idx + 1}/{len(TEST_QUESTIONS)}...")
        state = rag_graph.invoke({
            "query": q, "source_filter": None, "route": "",
            "context": [], "answer": "", "is_grounded": False, "retries": 0
        })
        context_str = "\n".join(state["context"])
        scores = score_both(q, context_str, state["answer"])

        results.append({
            "query": q,
            "retrieval_relevance": scores.get("retrieval_relevance", 0.0),
            "faithfulness": scores.get("faithfulness", 0.0),
            "grounded_flag": state["is_grounded"]
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_evaluation()