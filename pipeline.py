"""
Corrective Agentic RAG Pipeline — orchestrates the multi-agent workflow.

Implements the Corrective RAG architecture from the Agentic RAG survey paper:
  Step 1  Retrieve context from vector store
  Step 2  Evaluate relevance of each document (Reflection)
  Step 3  If context is weak → refine query and re-retrieve (Planning)
  Step 4  If still weak → fall back to web search (Tool Use)
  Step 5  Synthesize final response from validated documents
"""

from __future__ import annotations
from colorama import Fore, Style
from config import RELEVANCE_THRESHOLD, tracker
from agents import (
    retrieve_context,
    evaluate_relevance,
    refine_query,
    web_search,
    synthesize_response,
)

MAX_CORRECTION_ROUNDS = 2


def _header(text: str):
    print(f"\n{Fore.CYAN}{'-'*60}")
    print(f"  {text}")
    print(f"{'-'*60}{Style.RESET_ALL}")


def _step(label: str, detail: str = ""):
    print(f"  {Fore.YELLOW}[{label}]{Style.RESET_ALL} {detail}")


def run_pipeline(vectorstore, query: str) -> str:
    """Execute the full Corrective Agentic RAG pipeline."""
    corrections_log: list[str] = []

    _header(f"QUERY: {query}")

    # ── Step 1: Initial Retrieval ────────────────────────────────────────
    _step("Agent 1: Context Retrieval", "Searching vector knowledge base...")
    docs = retrieve_context(vectorstore, query)
    _step("Retrieved", f"{len(docs)} documents")
    for i, d in enumerate(docs):
        print(f"    {i+1}. [{d['source']}] {d['content'][:80]}...")

    # ── Step 2: Relevance Evaluation ─────────────────────────────────────
    _step("Agent 2: Relevance Evaluation", "Scoring document relevance...")
    all_docs, relevant, irrelevant = evaluate_relevance(query, docs)

    for d in all_docs:
        color = Fore.GREEN if d["relevance_score"] >= RELEVANCE_THRESHOLD else Fore.RED
        print(f"    {color}Score {d['relevance_score']:.2f}{Style.RESET_ALL} "
              f"[{d['source']}] {d['relevance_reason']}")

    # ── Step 3: Corrective Loop ──────────────────────────────────────────
    current_query = query
    for round_num in range(1, MAX_CORRECTION_ROUNDS + 1):
        if len(relevant) >= 2:
            _step("Evaluation", f"{Fore.GREEN}Sufficient relevant context ({len(relevant)} docs){Style.RESET_ALL}")
            break

        _step("Agent 3: Query Refinement", f"Round {round_num} — context insufficient, refining query...")
        feedback = "; ".join(d["relevance_reason"] for d in irrelevant)
        corrections_log.append(f"Round {round_num}: Refined query due to: {feedback}")

        current_query = refine_query(current_query, feedback)
        _step("Refined query", current_query)

        docs = retrieve_context(vectorstore, current_query)
        all_docs, relevant, irrelevant = evaluate_relevance(current_query, docs)

        for d in all_docs:
            color = Fore.GREEN if d["relevance_score"] >= RELEVANCE_THRESHOLD else Fore.RED
            print(f"    {color}Score {d['relevance_score']:.2f}{Style.RESET_ALL} "
                  f"[{d['source']}] {d['relevance_reason']}")

    # ── Step 4: Web Search Fallback ──────────────────────────────────────
    if len(relevant) < 2:
        _step("Agent 4: Web Search", "Falling back to web search for additional context...")
        web_docs = web_search(current_query)
        for wd in web_docs:
            wd["relevance_score"] = 0.7
            wd["relevance_reason"] = "Web search supplement"
        relevant.extend(web_docs)
        corrections_log.append("Web search fallback triggered")
        _step("Web results", f"Added {len(web_docs)} web documents")

    # ── Step 5: Response Synthesis ───────────────────────────────────────
    _step("Agent 5: Response Synthesis", "Generating final answer...")
    answer = synthesize_response(query, relevant, corrections_log)

    _header("FINAL ANSWER")
    print(f"\n{answer}\n")

    return answer
