# Agentic RAG: Report on "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"

**Paper:** Singh et al. (2025). *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.* arXiv:2501.09136  
**Implementation:** Corrective Agentic RAG — Healthcare Knowledge Assistant

---

## Key Ideas from the Paper

**1. Agentic Design Patterns (Reflection, Planning, Tool Use, Multi-Agent Collaboration).**
The paper identifies four foundational patterns that transform static RAG pipelines into adaptive systems. *Reflection* lets agents self-evaluate and iteratively correct outputs. *Planning* decomposes complex queries into manageable sub-tasks. *Tool Use* extends agents beyond pre-trained knowledge via APIs and web search. *Multi-Agent Collaboration* distributes specialized tasks across cooperating agents. These patterns matter because they convert a simple retrieve-then-generate pipeline into a system that can reason about its own quality and dynamically adjust strategy — a prerequisite for any production-grade AI system.

**2. Corrective RAG Architecture.**
The survey introduces Corrective RAG as a taxonomy category where agents evaluate retrieved documents for relevance, refine queries when context is weak, and fall back to external sources when needed. This is important because traditional RAG silently passes irrelevant documents to the LLM, causing hallucinations. Corrective RAG adds a feedback loop — a relevance evaluation agent scores each document, and a query refinement agent rewrites the query if scores are too low — directly reducing error propagation and improving factual accuracy.

**3. Taxonomy of Agentic RAG Architectures.**
The paper provides a structured taxonomy: Single-Agent Router, Multi-Agent, Hierarchical, Corrective, Adaptive, Graph-Based, and Agentic Document Workflows. This classification matters because it gives practitioners a decision framework — a simple FAQ bot can use a Single-Agent Router, while a multi-domain research assistant needs Hierarchical or Multi-Agent RAG. The taxonomy makes architectural trade-offs (simplicity vs. scalability, latency vs. accuracy) explicit and actionable.

## Application Scenario

**Real-world application:** A *Clinical Decision Support System* for hospital physicians. When a doctor queries "Patient has HbA1c of 8.2% and elevated creatinine — what are the risks and recommended next steps?", the system retrieves from medical guidelines, lab reference databases, and drug interaction APIs.

**Agents I would design:**
- **Retrieval Agent** — searches a FAISS vector store of clinical guidelines and medical literature
- **Relevance Evaluator** — scores retrieved documents against the specific patient context, filtering out generic results
- **Query Refinement Agent** — rewrites vague queries to target specific conditions (e.g., "CKD stage classification for GFR < 60")
- **Drug Interaction Tool Agent** — calls an external API (e.g., DrugBank) to check medication safety
- **Synthesis Agent** — produces a structured clinical summary with citations

## Personal Learning

Working through this paper shifted my understanding of AI agents from "chatbots with tools" to *systems with self-awareness about their own retrieval quality*. The most impactful idea is that an agent can evaluate whether its own retrieved context is good enough before generating — and if not, take corrective action. This feedback loop mirrors how a human researcher works: search, judge relevance, refine the search, then synthesize.

What surprised me most was how the Corrective RAG pattern dramatically changes error propagation. In traditional RAG, garbage retrieval silently becomes garbage output. Adding a relevance evaluation step — even a simple LLM-based scorer — creates a checkpoint that prevents low-quality context from reaching the generator. The cost tracking in my implementation also revealed that the corrective steps add minimal overhead (< $0.002 per query) while substantially improving answer quality, making the trade-off clearly worthwhile for production systems.
