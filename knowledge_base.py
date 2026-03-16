"""Builds and manages the vector knowledge base for healthcare documents."""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, tracker

HEALTHCARE_DOCS = [
    Document(
        page_content=(
            "Type 2 Diabetes is a chronic metabolic disorder characterized by insulin resistance "
            "and relative insulin deficiency. Common symptoms include increased thirst (polydipsia), "
            "frequent urination (polyuria), unexplained weight loss, fatigue, blurred vision, and "
            "slow-healing wounds. Risk factors include obesity, sedentary lifestyle, family history, "
            "and age over 45. HbA1c levels above 6.5% confirm diagnosis."
        ),
        metadata={"source": "medical_guidelines", "topic": "diabetes"},
    ),
    Document(
        page_content=(
            "Cardiovascular disease (CVD) encompasses conditions affecting the heart and blood vessels, "
            "including coronary artery disease, heart failure, and stroke. Major risk factors include "
            "hypertension, high cholesterol, smoking, diabetes, obesity, and physical inactivity. "
            "Patients with Type 2 Diabetes have a 2-4x higher risk of cardiovascular events. "
            "Shared pathways include chronic inflammation, endothelial dysfunction, and oxidative stress."
        ),
        metadata={"source": "cardiology_review", "topic": "cardiovascular"},
    ),
    Document(
        page_content=(
            "Hypertension (high blood pressure) is a leading modifiable risk factor for cardiovascular "
            "disease and stroke. Blood pressure consistently above 130/80 mmHg is classified as hypertension. "
            "First-line treatments include ACE inhibitors, ARBs, calcium channel blockers, and thiazide "
            "diuretics. Lifestyle modifications such as DASH diet, reduced sodium intake, regular exercise, "
            "and weight management are essential adjuncts to pharmacotherapy."
        ),
        metadata={"source": "hypertension_guidelines", "topic": "hypertension"},
    ),
    Document(
        page_content=(
            "Mental health disorders, particularly depression and anxiety, are increasingly recognized "
            "as comorbidities in patients with chronic diseases like diabetes and heart disease. "
            "Depression affects 15-25% of diabetes patients — double the rate of the general population. "
            "Bidirectional mechanisms include HPA axis dysregulation, inflammatory cytokines, and "
            "behavioral factors such as poor medication adherence and unhealthy diet."
        ),
        metadata={"source": "psychiatry_journal", "topic": "mental_health"},
    ),
    Document(
        page_content=(
            "Preventive medicine emphasizes vaccination, screening, and lifestyle counseling to reduce "
            "disease burden. Recommended screenings for adults over 40 include fasting glucose or HbA1c "
            "for diabetes, lipid panel for cardiovascular risk, colonoscopy for colorectal cancer, and "
            "blood pressure measurement at every clinical visit. Early detection allows intervention "
            "before complications develop, significantly improving patient outcomes."
        ),
        metadata={"source": "preventive_care_manual", "topic": "preventive_medicine"},
    ),
    Document(
        page_content=(
            "Chronic Kidney Disease (CKD) is a progressive loss of kidney function over months or years. "
            "Diabetes and hypertension account for nearly two-thirds of all CKD cases. Stages are classified "
            "by glomerular filtration rate (GFR): Stage 1 (GFR >= 90) through Stage 5 (GFR < 15, requiring "
            "dialysis or transplant). Management includes blood pressure control with ACE inhibitors, "
            "glycemic control, dietary protein restriction, and avoiding nephrotoxic medications."
        ),
        metadata={"source": "nephrology_textbook", "topic": "kidney_disease"},
    ),
]


def build_knowledge_base() -> FAISS:
    """Chunk documents and create a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(HEALTHCARE_DOCS)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    total_chars = sum(len(c.page_content) for c in chunks)
    tracker.add_embedding_usage(total_chars // 4)
    tracker.retrieval_calls += 1

    print(f"  [KnowledgeBase] Indexed {len(chunks)} chunks from {len(HEALTHCARE_DOCS)} documents")
    return vectorstore
