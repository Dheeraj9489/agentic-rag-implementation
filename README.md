# Agentic RAG -- Healthcare Knowledge Assistant

A **Corrective Agentic RAG** system implementing the multi-agent architecture described in
[*"Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"*](https://arxiv.org/abs/2501.09136)
(Singh et al., 2025).

Supports **OpenAI** and **Ollama (local)** as LLM providers.

Forked from: [asinghcsu/AgenticRAG-Survey](https://github.com/asinghcsu/AgenticRAG-Survey)

---

## Architecture

The system implements five specialized agents arranged in a Corrective RAG pipeline:

```
User Query
    |
    v
[Agent 1: Context Retrieval]     -- FAISS vector search over healthcare documents
    |
    v
[Agent 2: Relevance Evaluation]  -- LLM scores each document (Reflection pattern)
    |
    v  (score < threshold?)
[Agent 3: Query Refinement]      -- Rewrites query for better retrieval (Planning)
    |
    v  (still insufficient?)
[Agent 4: Web Search]            -- DuckDuckGo fallback (Tool Use pattern)
    |
    v
[Agent 5: Response Synthesis]    -- Final answer with source citations
```

### Agentic Patterns Demonstrated

| Pattern | Agent | Description |
|---------|-------|-------------|
| **Reflection** | Relevance Evaluator | Self-evaluates retrieved document quality |
| **Planning** | Query Refinement | Decomposes and rewrites queries when context is weak |
| **Tool Use** | Web Search | Falls back to external web search via DuckDuckGo |
| **Multi-Agent** | All | Five agents collaborate in a corrective pipeline |

---

## LLM Provider Support

| Provider | LLM | Embeddings | Cost |
|----------|-----|------------|------|
| **OpenAI** | gpt-4o-mini | text-embedding-3-small | Pay-per-token |
| **Ollama** | llama3 (configurable) | nomic-embed-text (configurable) | Free (local) |

The provider is selected via the `--ollama` CLI flag, the `LLM_PROVIDER` env var, or auto-detected from the presence of `OPENAI_API_KEY`.

---

## Quick Start

### 1. Clone and set up virtual environment

```bash
git clone https://github.com/Dheeraj9489/agentic-rag-implementation.git
cd agentic-rag-implementation
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in demo mode (no API key or model needed)

```bash
python main.py --demo
```

### 4. Run with OpenAI

```bash
# Create .env with your key
echo OPENAI_API_KEY=sk-your-key-here > .env

python main.py
```

### 5. Run with Ollama (local, free)

```bash
# Install Ollama: https://ollama.com/download
# Pull a model and an embedding model:
ollama pull llama3
ollama pull nomic-embed-text

# Run the system:
python main.py --ollama
```

You can customize the Ollama model via environment variables or `.env`:

```bash
OLLAMA_LLM_MODEL=mistral              # default: llama3
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large  # default: nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434 # default
```

### 6. Custom queries

```bash
# Works with any provider
python main.py "What treatments exist for hypertension in diabetic patients?"
python main.py --ollama "What are the risk factors for chronic kidney disease?"
```

---

## Project Structure

```
agentic-rag-implementation/
  main.py            # Entry point (--demo, --ollama, or auto-detect OpenAI)
  config.py          # Provider config, model settings, cost tracking
  knowledge_base.py  # FAISS vector store with healthcare documents
  agents.py          # Five specialized agents (provider-agnostic)
  pipeline.py        # Corrective RAG orchestration pipeline
  demo_mode.py       # Simulation mode (no API key or model required)
  requirements.txt   # Python dependencies
  REPORT.md          # 1-page analysis report
  .env.example       # Template for provider configuration
  .gitignore         # Excludes .venv, .env, __pycache__
  screenshots/       # Run output captures
```

---

## Sample Run Output

### Demo 1: Diabetes and Heart Disease (Direct retrieval -- no correction needed)

```
QUERY: What are the common symptoms of Type 2 Diabetes and how is it related to heart disease?

[Agent 1: Context Retrieval] Searching vector knowledge base...
[Retrieved] 3 documents

[Agent 2: Relevance Evaluation] Scoring document relevance...
  Score 0.95 [medical_guidelines] Directly addresses Type 2 Diabetes symptoms.
  Score 0.90 [cardiology_review] Explains the cardiovascular-diabetes link.
  Score 0.35 [psychiatry_journal] Discusses mental health, not heart disease.

[Evaluation] Sufficient relevant context (2 docs)
[Agent 5: Response Synthesis] Generating final answer...

FINAL ANSWER:
Type 2 Diabetes symptoms include increased thirst, frequent urination,
fatigue, and blurred vision. Patients have 2-4x higher cardiovascular
risk due to chronic inflammation and oxidative stress.
```

### Demo 2: Preventive Screening (Corrective loop triggered)

```
QUERY: What screening tests should adults over 40 get for preventive care?

[Agent 2: Relevance Evaluation]
  Score 0.92 [preventive_care_manual] -- relevant
  Score 0.40 [medical_guidelines]    -- irrelevant
  Score 0.38 [cardiology_review]     -- irrelevant

[Agent 3: Query Refinement] Only 1 relevant doc, refining query...
  Refined: "What specific preventive screening tests are recommended
            for adults aged 40 and above?"

  Score 0.95 [preventive_care_manual] -- relevant
  Score 0.65 [nephrology_textbook]    -- relevant (CKD screening)

[Agent 5: Response Synthesis] Generating final answer...
```

### Cost Report

```
============================================================
  COST & USAGE REPORT  (provider: openai)
============================================================
  LLM calls:            16
  Retrieval calls:      5
  Web searches:         0
  Corrections applied:  1
  Prompt tokens:        7,400
  Completion tokens:    1,010
  Estimated cost:       $0.001726
  Wall-clock time:      3.31s
============================================================
```

When using Ollama, the cost report shows `$0.00 (local model)` since inference is free.

---

## Report

See [REPORT.md](REPORT.md) for the full analysis covering:
- Three key ideas from the paper
- Application scenario (Clinical Decision Support System)
- Personal learning reflections

---

## References

- Singh, A., Ehtesham, A., Kumar, S., & Talaei Khoei, T. (2025). *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.* arXiv:2501.09136. https://arxiv.org/abs/2501.09136
- Original survey repo: https://github.com/asinghcsu/AgenticRAG-Survey
