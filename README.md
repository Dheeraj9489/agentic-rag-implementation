# Agentic RAG -- Healthcare Knowledge Assistant

A **Corrective Agentic RAG** system implementing the multi-agent architecture described in
[*"Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"*](https://arxiv.org/abs/2501.09136)
(Singh et al., 2025).

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

## Quick Start

### 1. Clone and set up virtual environment

```bash
git clone <your-repo-url>
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

### 3. Run in demo mode (no API key needed)

```bash
python main.py --demo
```

### 4. Run with OpenAI API (full live mode)

```bash
# Create .env with your key
echo OPENAI_API_KEY=sk-your-key-here > .env

python main.py
```

### 5. Custom queries

```bash
python main.py "What treatments exist for hypertension in diabetic patients?"
```

---

## Project Structure

```
agentic-rag-implementation/
  main.py            # Entry point (auto-detects API key or uses demo mode)
  config.py          # Configuration, cost tracking
  knowledge_base.py  # FAISS vector store with healthcare documents
  agents.py          # Five specialized agents
  pipeline.py        # Corrective RAG orchestration pipeline
  demo_mode.py       # Simulation mode (no API key required)
  requirements.txt   # Python dependencies
  REPORT.md          # 1-page analysis report
  .env.example       # Template for API key
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
  COST & USAGE REPORT
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
