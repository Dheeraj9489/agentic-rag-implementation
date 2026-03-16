"""
Agentic RAG — Healthcare Knowledge Assistant

Demonstrates Corrective Agentic RAG with five specialized agents:
  1. Context Retrieval Agent       (vector search)
  2. Relevance Evaluation Agent    (reflection / self-critique)
  3. Query Refinement Agent        (planning / corrective rewrite)
  4. Web Search Agent              (tool use fallback)
  5. Response Synthesis Agent      (final answer generation)

Based on: "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"
          (Singh et al., 2025) — arXiv:2501.09136

Usage:
  python main.py                          # run with API key from .env
  python main.py --demo                   # run in demo/simulation mode
  python main.py "your custom query"      # run a custom query
"""

import sys
import os
from colorama import init as colorama_init, Fore, Style
from dotenv import load_dotenv

load_dotenv()


def run_live():
    """Run full pipeline with OpenAI API."""
    from config import tracker
    from knowledge_base import build_knowledge_base
    from pipeline import run_pipeline

    DEMO_QUERIES = [
        "What are the common symptoms of Type 2 Diabetes and how is it related to heart disease?",
        "What screening tests should adults over 40 get for preventive care?",
        "How does chronic kidney disease relate to diabetes and what treatments are available?",
    ]

    print(f"\n{Fore.MAGENTA}{'='*60}")
    print("  AGENTIC RAG — Healthcare Knowledge Assistant")
    print("  Corrective RAG with Multi-Agent Architecture")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}Building knowledge base...{Style.RESET_ALL}")
    vectorstore = build_knowledge_base()
    print()

    args = [a for a in sys.argv[1:] if a != "--demo"]
    queries = args if args else DEMO_QUERIES

    for i, query in enumerate(queries):
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"  DEMO {i+1} of {len(queries)}")
        print(f"{'='*60}{Style.RESET_ALL}")
        run_pipeline(vectorstore, query)

    print(tracker.summary())


def run_demo():
    """Run simulated pipeline without API key."""
    from demo_mode import main as demo_main
    demo_main()


def main():
    colorama_init()

    use_demo = "--demo" in sys.argv
    has_key = bool(os.getenv("OPENAI_API_KEY"))

    if use_demo or not has_key:
        if not has_key and not use_demo:
            print(f"\n{Fore.YELLOW}  No OPENAI_API_KEY found. Running in demo mode.")
            print(f"  Set OPENAI_API_KEY in .env for live mode.{Style.RESET_ALL}")
        run_demo()
    else:
        run_live()


if __name__ == "__main__":
    main()
