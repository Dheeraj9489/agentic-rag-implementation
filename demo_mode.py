"""
Simulation mode — demonstrates the Corrective Agentic RAG pipeline
without requiring an OpenAI API key.

Uses pre-computed results to show the full multi-agent workflow,
including retrieval, relevance evaluation, query refinement, and synthesis.
"""

from __future__ import annotations
import time
from colorama import Fore, Style
from config import tracker

SIMULATED_KNOWLEDGE = {
    "diabetes": {
        "content": (
            "Type 2 Diabetes is a chronic metabolic disorder characterized by insulin resistance. "
            "Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. "
            "Risk factors include obesity, sedentary lifestyle, and age over 45. HbA1c > 6.5% confirms diagnosis."
        ),
        "source": "medical_guidelines",
        "topic": "diabetes",
    },
    "cardiovascular": {
        "content": (
            "Cardiovascular disease encompasses coronary artery disease, heart failure, and stroke. "
            "Major risk factors include hypertension, high cholesterol, smoking, diabetes, and obesity. "
            "Patients with Type 2 Diabetes have a 2-4x higher risk of cardiovascular events due to "
            "chronic inflammation, endothelial dysfunction, and oxidative stress."
        ),
        "source": "cardiology_review",
        "topic": "cardiovascular",
    },
    "preventive": {
        "content": (
            "Preventive medicine recommends screenings for adults over 40: fasting glucose/HbA1c for diabetes, "
            "lipid panel for cardiovascular risk, colonoscopy for colorectal cancer, and blood pressure checks. "
            "Early detection significantly improves patient outcomes."
        ),
        "source": "preventive_care_manual",
        "topic": "preventive_medicine",
    },
    "kidney": {
        "content": (
            "Chronic Kidney Disease (CKD) is progressive loss of kidney function. Diabetes and hypertension "
            "account for nearly two-thirds of CKD cases. Management includes blood pressure control with "
            "ACE inhibitors, glycemic control, dietary protein restriction, and avoiding nephrotoxic medications."
        ),
        "source": "nephrology_textbook",
        "topic": "kidney_disease",
    },
    "mental_health": {
        "content": (
            "Depression affects 15-25% of diabetes patients, double the general population rate. "
            "Bidirectional mechanisms include HPA axis dysregulation, inflammatory cytokines, and "
            "behavioral factors such as poor medication adherence."
        ),
        "source": "psychiatry_journal",
        "topic": "mental_health",
    },
}

QUERY_RESPONSES = {
    0: {
        "retrieved": ["diabetes", "cardiovascular", "mental_health"],
        "relevant": ["diabetes", "cardiovascular"],
        "irrelevant": ["mental_health"],
        "scores": {"diabetes": 0.95, "cardiovascular": 0.90, "mental_health": 0.35},
        "reasons": {
            "diabetes": "Directly addresses Type 2 Diabetes symptoms asked in the query.",
            "cardiovascular": "Explains the cardiovascular-diabetes link with shared risk factors.",
            "mental_health": "Discusses mental health comorbidities, not directly about heart disease symptoms.",
        },
        "answer": (
            "**Type 2 Diabetes Symptoms:**\n"
            "Common symptoms include increased thirst (polydipsia), frequent urination (polyuria), "
            "unexplained weight loss, fatigue, blurred vision, and slow-healing wounds. Diagnosis is "
            "confirmed by HbA1c levels above 6.5% [Source: medical_guidelines].\n\n"
            "**Relationship to Heart Disease:**\n"
            "Patients with Type 2 Diabetes have a 2-4x higher risk of cardiovascular events. The connection "
            "is driven by shared pathways including chronic inflammation, endothelial dysfunction, and "
            "oxidative stress. Common risk factors such as obesity and hypertension further compound this "
            "relationship [Source: cardiology_review]."
        ),
    },
    1: {
        "retrieved": ["preventive", "diabetes", "cardiovascular"],
        "relevant": ["preventive"],
        "irrelevant": ["diabetes", "cardiovascular"],
        "scores": {"preventive": 0.92, "diabetes": 0.40, "cardiovascular": 0.38},
        "reasons": {
            "preventive": "Directly lists recommended screenings for adults over 40.",
            "diabetes": "Discusses diabetes symptoms, not screening recommendations.",
            "cardiovascular": "Focuses on CVD pathology, not preventive screening.",
        },
        "refined_query": "What specific preventive screening tests and health checks are recommended for adults aged 40 and above?",
        "refined_retrieved": ["preventive", "kidney", "diabetes"],
        "refined_relevant": ["preventive", "kidney"],
        "refined_scores": {"preventive": 0.95, "kidney": 0.65, "diabetes": 0.45},
        "refined_reasons": {
            "preventive": "Comprehensive list of recommended screenings for adults over 40.",
            "kidney": "CKD screening is relevant for adults over 40 with risk factors.",
            "diabetes": "Discusses diabetes in general, not specific screening protocols.",
        },
        "answer": (
            "**Recommended Screenings for Adults Over 40:**\n\n"
            "1. **Diabetes Screening:** Fasting glucose or HbA1c test [Source: preventive_care_manual]\n"
            "2. **Cardiovascular Risk:** Lipid panel (cholesterol levels) [Source: preventive_care_manual]\n"
            "3. **Colorectal Cancer:** Colonoscopy [Source: preventive_care_manual]\n"
            "4. **Hypertension:** Blood pressure measurement at every clinical visit [Source: preventive_care_manual]\n"
            "5. **Kidney Function:** GFR and urine albumin tests, especially for patients with diabetes or "
            "hypertension [Source: nephrology_textbook]\n\n"
            "Early detection through these screenings allows intervention before complications develop, "
            "significantly improving patient outcomes."
        ),
    },
    2: {
        "retrieved": ["kidney", "diabetes", "cardiovascular"],
        "relevant": ["kidney", "diabetes"],
        "irrelevant": ["cardiovascular"],
        "scores": {"kidney": 0.95, "diabetes": 0.72, "cardiovascular": 0.30},
        "reasons": {
            "kidney": "Directly addresses CKD, its link to diabetes, and available treatments.",
            "diabetes": "Provides context on diabetes as a leading cause of CKD.",
            "cardiovascular": "Focuses on cardiovascular disease, not kidney disease.",
        },
        "answer": (
            "**Chronic Kidney Disease (CKD) and Diabetes:**\n"
            "Diabetes and hypertension account for nearly two-thirds of all CKD cases. CKD is classified "
            "by GFR stages: Stage 1 (GFR >= 90) through Stage 5 (GFR < 15, requiring dialysis or transplant) "
            "[Source: nephrology_textbook].\n\n"
            "**Available Treatments:**\n"
            "- Blood pressure control with ACE inhibitors\n"
            "- Glycemic control (HbA1c management)\n"
            "- Dietary protein restriction\n"
            "- Avoiding nephrotoxic medications\n"
            "[Source: nephrology_textbook]\n\n"
            "**Why Diabetes Matters:**\n"
            "The insulin resistance and chronic hyperglycemia in Type 2 Diabetes damage kidney blood vessels "
            "over time, making glycemic control (HbA1c < 6.5%) critical for CKD prevention [Source: medical_guidelines]."
        ),
    },
}


def _header(text: str):
    print(f"\n{Fore.CYAN}{'-'*60}")
    print(f"  {text}")
    print(f"{'-'*60}{Style.RESET_ALL}")


def _step(label: str, detail: str = ""):
    print(f"  {Fore.YELLOW}[{label}]{Style.RESET_ALL} {detail}")


def _pause(seconds: float = 0.3):
    time.sleep(seconds)


def run_demo(query_index: int, query: str):
    """Run the simulated Corrective Agentic RAG pipeline for a given query."""
    data = QUERY_RESPONSES[query_index]

    _header(f"QUERY: {query}")

    # ── Step 1: Retrieval ────────────────────────────────────────────────
    _step("Agent 1: Context Retrieval", "Searching vector knowledge base...")
    _pause()
    tracker.retrieval_calls += 1
    docs = data["retrieved"]
    _step("Retrieved", f"{len(docs)} documents")
    for i, key in enumerate(docs):
        d = SIMULATED_KNOWLEDGE[key]
        print(f"    {i+1}. [{d['source']}] {d['content'][:80]}...")

    # ── Step 2: Relevance Evaluation ─────────────────────────────────────
    _step("Agent 2: Relevance Evaluation", "Scoring document relevance...")
    _pause()
    tracker.llm_calls += len(docs)
    tracker.prompt_tokens += 450 * len(docs)
    tracker.completion_tokens += 30 * len(docs)

    for key in docs:
        score = data["scores"][key]
        reason = data["reasons"][key]
        color = Fore.GREEN if score >= 0.6 else Fore.RED
        print(f"    {color}Score {score:.2f}{Style.RESET_ALL} "
              f"[{SIMULATED_KNOWLEDGE[key]['source']}] {reason}")

    relevant_keys = data["relevant"]
    irrelevant_keys = data["irrelevant"]

    # ── Step 3: Corrective Loop ──────────────────────────────────────────
    if "refined_query" in data:
        _step("Agent 3: Query Refinement",
              f"Round 1 — only {len(relevant_keys)} relevant doc(s), refining query...")
        _pause()
        tracker.llm_calls += 1
        tracker.prompt_tokens += 200
        tracker.completion_tokens += 50
        tracker.corrections_applied += 1

        _step("Refined query", data["refined_query"])
        _pause()

        tracker.retrieval_calls += 1
        tracker.llm_calls += 3
        tracker.prompt_tokens += 450 * 3
        tracker.completion_tokens += 30 * 3

        for key in data["refined_retrieved"]:
            score = data["refined_scores"][key]
            reason = data["refined_reasons"][key]
            color = Fore.GREEN if score >= 0.6 else Fore.RED
            print(f"    {color}Score {score:.2f}{Style.RESET_ALL} "
                  f"[{SIMULATED_KNOWLEDGE[key]['source']}] {reason}")

        relevant_keys = data["refined_relevant"]
    else:
        _step("Evaluation",
              f"{Fore.GREEN}Sufficient relevant context ({len(relevant_keys)} docs){Style.RESET_ALL}")

    # ── Step 5: Synthesis ────────────────────────────────────────────────
    _step("Agent 5: Response Synthesis", "Generating final answer...")
    _pause()
    tracker.llm_calls += 1
    tracker.prompt_tokens += 600
    tracker.completion_tokens += 200

    _header("FINAL ANSWER")
    print(f"\n{data['answer']}\n")

    return data["answer"]


DEMO_QUERIES = [
    "What are the common symptoms of Type 2 Diabetes and how is it related to heart disease?",
    "What screening tests should adults over 40 get for preventive care?",
    "How does chronic kidney disease relate to diabetes and what treatments are available?",
]


def main():
    from colorama import init as colorama_init
    colorama_init()

    print(f"\n{Fore.MAGENTA}{'='*60}")
    print("  AGENTIC RAG — Healthcare Knowledge Assistant")
    print("  Corrective RAG with Multi-Agent Architecture")
    print(f"  [DEMO MODE — no API key required]")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}Building knowledge base (simulated)...{Style.RESET_ALL}")
    print(f"  [KnowledgeBase] Indexed 8 chunks from 6 documents\n")
    tracker.retrieval_calls += 1
    tracker.embedding_tokens += 500

    for i, query in enumerate(DEMO_QUERIES):
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"  DEMO {i+1} of {len(DEMO_QUERIES)}")
        print(f"{'='*60}{Style.RESET_ALL}")
        run_demo(i, query)

    print(tracker.summary())


if __name__ == "__main__":
    main()
