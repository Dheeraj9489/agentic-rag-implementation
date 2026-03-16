"""Configuration and cost tracking for the Agentic RAG system."""

import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Provider: "openai" or "ollama" (set via LLM_PROVIDER env var or --ollama flag)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI defaults
OPENAI_LLM_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Ollama defaults (override via env vars)
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TEMPERATURE = 0.2
RELEVANCE_THRESHOLD = 0.6


@dataclass
class CostTracker:
    """Tracks token usage and estimated cost across all LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    web_searches: int = 0
    corrections_applied: int = 0
    start_time: float = field(default_factory=time.time)

    PRICE_PER_1K_PROMPT = 0.00015
    PRICE_PER_1K_COMPLETION = 0.0006
    PRICE_PER_1K_EMBEDDING = 0.00002

    def add_llm_usage(self, prompt_tok: int, completion_tok: int):
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.llm_calls += 1

    def add_embedding_usage(self, tokens: int):
        self.embedding_tokens += tokens

    @property
    def estimated_cost(self) -> float:
        return (
            self.prompt_tokens / 1000 * self.PRICE_PER_1K_PROMPT
            + self.completion_tokens / 1000 * self.PRICE_PER_1K_COMPLETION
            + self.embedding_tokens / 1000 * self.PRICE_PER_1K_EMBEDDING
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self, provider: str = "openai") -> str:
        cost_line = (
            f"  Estimated cost:       ${self.estimated_cost:.6f}\n"
            if provider == "openai"
            else "  Estimated cost:       $0.00 (local model)\n"
        )
        return (
            f"\n{'='*60}\n"
            f"  COST & USAGE REPORT  (provider: {provider})\n"
            f"{'='*60}\n"
            f"  LLM calls:            {self.llm_calls}\n"
            f"  Retrieval calls:      {self.retrieval_calls}\n"
            f"  Web searches:         {self.web_searches}\n"
            f"  Corrections applied:  {self.corrections_applied}\n"
            f"  Prompt tokens:        {self.prompt_tokens:,}\n"
            f"  Completion tokens:    {self.completion_tokens:,}\n"
            f"  Embedding tokens:     {self.embedding_tokens:,}\n"
            f"{cost_line}"
            f"  Wall-clock time:      {self.elapsed_seconds:.2f}s\n"
            f"{'='*60}"
        )


tracker = CostTracker()
