"""
Specialized agents implementing Agentic RAG patterns from the survey paper.

Agents:
  1. ContextRetrievalAgent   — retrieves documents from the vector store
  2. RelevanceEvaluationAgent — scores document relevance (Corrective RAG reflection)
  3. QueryRefinementAgent    — rewrites queries when retrieved context is weak
  4. WebSearchAgent          — falls back to live web search (Tool Use pattern)
  5. ResponseSynthesisAgent  — generates the final answer from validated context
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS
from config import (
    LLM_PROVIDER, TEMPERATURE, RELEVANCE_THRESHOLD, tracker,
    OPENAI_LLM_MODEL, OLLAMA_LLM_MODEL, OLLAMA_BASE_URL,
)


def _llm():
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=TEMPERATURE)


def _track(response):
    usage = response.response_metadata.get("token_usage", {})
    tracker.add_llm_usage(
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


# ── Agent 1: Context Retrieval ──────────────────────────────────────────────

def retrieve_context(vectorstore, query: str, k: int = 3) -> list[dict]:
    """Retrieve top-k documents from the FAISS vector store."""
    docs = vectorstore.similarity_search_with_score(query, k=k)
    tracker.retrieval_calls += 1
    results = []
    for doc, score in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "topic": doc.metadata.get("topic", "unknown"),
            "similarity": round(1 - score / 100, 4),  # normalize
        })
    return results


# ── Agent 2: Relevance Evaluation (Reflection pattern) ──────────────────────

def evaluate_relevance(query: str, documents: list[dict]) -> list[dict]:
    """Score each retrieved document for relevance to the query."""
    llm = _llm()
    evaluated = []

    for doc in documents:
        prompt = (
            f"Rate the relevance of the following document to the query on a scale of 0.0-1.0.\n"
            f"Reply with ONLY a JSON object: {{\"score\": <float>, \"reason\": \"<1 sentence>\"}}\n\n"
            f"Query: {query}\n\n"
            f"Document: {doc['content']}"
        )
        response = llm.invoke([
            SystemMessage(content="You are a relevance evaluation agent. Be strict and precise."),
            HumanMessage(content=prompt),
        ])
        _track(response)

        import json
        try:
            result = json.loads(response.content.strip().strip("```json").strip("```"))
            doc["relevance_score"] = result["score"]
            doc["relevance_reason"] = result["reason"]
        except (json.JSONDecodeError, KeyError):
            doc["relevance_score"] = 0.5
            doc["relevance_reason"] = "Could not parse evaluation"

        evaluated.append(doc)

    relevant = [d for d in evaluated if d["relevance_score"] >= RELEVANCE_THRESHOLD]
    irrelevant = [d for d in evaluated if d["relevance_score"] < RELEVANCE_THRESHOLD]

    return evaluated, relevant, irrelevant


# ── Agent 3: Query Refinement (Planning pattern) ────────────────────────────

def refine_query(original_query: str, feedback: str) -> str:
    """Rewrite the query to improve retrieval quality."""
    llm = _llm()
    prompt = (
        f"The original query did not retrieve sufficiently relevant results.\n"
        f"Original query: {original_query}\n"
        f"Feedback: {feedback}\n\n"
        f"Rewrite the query to be more specific and likely to retrieve relevant medical information. "
        f"Reply with ONLY the rewritten query text."
    )
    response = llm.invoke([
        SystemMessage(content="You are a query refinement agent specializing in healthcare queries."),
        HumanMessage(content=prompt),
    ])
    _track(response)
    tracker.corrections_applied += 1
    return response.content.strip()


# ── Agent 4: Web Search (Tool Use pattern) ──────────────────────────────────

def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web using DuckDuckGo as a fallback knowledge source."""
    tracker.web_searches += 1
    try:
        results = DDGS().text(query, max_results=max_results)
        return [
            {"content": r.get("body", ""), "source": r.get("href", "web"), "topic": "web_search"}
            for r in results
        ]
    except Exception as e:
        return [{"content": f"Web search failed: {e}", "source": "error", "topic": "web_search"}]


# ── Agent 5: Response Synthesis ─────────────────────────────────────────────

def synthesize_response(query: str, context_docs: list[dict], corrections_log: list[str]) -> str:
    """Generate a final response from validated context documents."""
    llm = _llm()

    context_text = "\n\n".join(
        f"[Source: {d['source']}] {d['content']}" for d in context_docs
    )
    corrections_text = "\n".join(corrections_log) if corrections_log else "None"

    prompt = (
        f"Answer the following healthcare question using ONLY the provided context.\n"
        f"If the context is insufficient, say so. Cite sources inline.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Corrections applied during retrieval:\n{corrections_text}"
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are a medical knowledge synthesis agent. Provide accurate, well-structured "
            "answers with source citations. Do not fabricate information."
        )),
        HumanMessage(content=prompt),
    ])
    _track(response)
    return response.content.strip()
