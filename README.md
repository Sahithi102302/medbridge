# MedBridge — AI-Powered Medical Document Translator

> Translating complex medical documents into plain English for patients.

**Live Demo:** [medbridge-blmgb4sam-sahithi-projects1.vercel.app](https://medbridge-blmgb4sam-sahithi-projects1.vercel.app)  
**Backend API:** [web-production-6a16b1.up.railway.app](https://web-production-6a16b1.up.railway.app)  
**GitHub:** [github.com/Sahithi102302/medbridge](https://github.com/Sahithi102302/medbridge)

---

## The Problem

When patients leave a hospital, they receive discharge papers, lab reports, radiology reports, and insurance documents filled with medical jargon most people cannot understand. Studies show that over 40% of patients do not understand their own discharge instructions, leading to medication errors, missed follow-ups, and preventable readmissions.

MedBridge solves this by taking any medical PDF and returning:
- A plain English summary of what happened
- Every medication explained — what it does and why this patient needs it
- A glossary of medical terms with source sentence tracing
- Urgency flags — what to do and when
- Five follow-up questions to bring to the next doctor's appointment

---

## Live Demo

Upload any of the following document types:
- **Discharge Summary** — post-hospitalization instructions
- **Lab Report** — blood work and test results
- **Radiology Report** — X-ray, CT, MRI findings
- **Insurance EOB** — Explanation of Benefits

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    5-Stage Pipeline                      │
│                                                         │
│  1. PDF Parser        PyMuPDF                           │
│     Extract clean text from any medical PDF             │
│                          │                              │
│  2. Classifier        TF-IDF + Logistic Regression      │
│     Detect document type (discharge/lab/radiology/eob)  │
│                          │                              │
│  3. NER               scispaCy BC5CDR                   │
│     Extract DISEASE and CHEMICAL entities               │
│                          │                              │
│  4. Chunker           Regex section splitter            │
│     Split into sections, enforce 6000 token budget      │
│                          │                              │
│  5. LLM Translation   GPT-4o-mini / Gemini 2.5 Flash   │
│     Translate to plain English with grounding rules     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Structured Output (Pydantic validated)
    │
    ├── Summary
    ├── Urgency Flags
    ├── Medications
    ├── Jargon Glossary
    └── Doctor Questions
    │
    ▼
FastAPI Backend → React Frontend
```

---

## Evaluation Results

Evaluated across **10 real de-identified MIMIC-IV clinical notes**:

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Flesch-Kincaid Readability Grade | 4.37 | 2–8 | ✅ PASS |
| NER Grounding Rate | 100% | ≥ 90% | ✅ PASS |
| LLM-as-Judge Clarity | 5.0 / 5 | ≥ 4.0 | ✅ PASS |
| LLM-as-Judge Accuracy | 4.7 / 5 | ≥ 4.0 | ✅ PASS |
| LLM-as-Judge Overall | 4.23 / 5 | ≥ 4.0 | ✅ PASS |
| Pipeline Errors | 0 / 10 | 0 | ✅ PASS |

**Dataset:** MIMIC-IV-Ext-BHC — 270,033 real de-identified hospital discharge notes from Beth Israel Deaconess Medical Center (MIT, 2025). Access requires PhysioNet credentials and CITI research ethics certification.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| PDF Parsing | PyMuPDF | Extract clean text from medical PDFs |
| Classification | scikit-learn TF-IDF + LogReg | Detect document type |
| Medical NER | scispaCy BC5CDR | Extract DISEASE and CHEMICAL entities |
| LLM Translation | GPT-4o-mini / Gemini 2.5 Flash | Plain English translation |
| Output Validation | Pydantic | Enforce structured JSON output |
| Backend | FastAPI + uvicorn | REST API with SSE streaming |
| Frontend | React + Vite + Tailwind CSS | User interface |
| Backend Hosting | Railway | Cloud deployment |
| Frontend Hosting | Vercel | CDN deployment |
| Evaluation | textstat + Groq Qwen | Readability + LLM judge |
| Experiment Tracking | MLflow | Metrics logging |
| Dataset | MIMIC-IV-Ext-BHC | Real clinical notes |

---

## Project Structure

```
medbridge/
│
├── src/medbridge/
│   ├── ingest/
│   │   └── parser.py              # PyMuPDF PDF extraction
│   ├── nlp/
│   │   ├── classifier.py          # TF-IDF + LogReg document classifier
│   │   ├── ner.py                 # scispaCy BC5CDR medical NER
│   │   └── chunker.py             # Section splitter with token budget
│   ├── llm/
│   │   ├── schemas.py             # Pydantic output models
│   │   ├── prompt_templates.py    # 4 document-type-specific prompts
│   │   └── client.py             # LLM API client with retry logic
│   ├── eval/
│   │   ├── readability.py         # Flesch-Kincaid scoring
│   │   ├── grounding.py           # NER hallucination detection
│   │   └── judge.py               # LLM-as-judge evaluation
│   └── pipeline.py                # Main orchestrator
│
├── api/
│   └── main.py                    # FastAPI endpoints
│
├── frontend/
│   └── src/
│       ├── App.jsx                # Main app with SSE handling
│       └── components/
│           ├── UploadScreen.jsx   # Drag-and-drop PDF upload
│           ├── ProcessingScreen.jsx # Live pipeline progress
│           ├── ResultsDashboard.jsx # 4-tab results view
│           ├── UrgencyPanel.jsx   # Critical action flags
│           ├── MedicationList.jsx # Drug cards with warnings
│           ├── JargonTab.jsx      # Medical term glossary
│           └── QuestionsTab.jsx   # Doctor questions with copy
│
├── data/
│   ├── models/                    # Trained classifier artifacts
│   │   ├── tfidf_vectorizer.joblib
│   │   └── doc_classifier.joblib
│   └── samples/                   # Test PDFs (MIMIC CSV excluded)
│
├── notebooks/
│   └── 01_eda_and_classifier_training.ipynb
│
├── run_eval.py                    # Evaluation harness runner
├── requirements.txt
├── railway.toml
├── Procfile
└── .env.example
```

---

## Key Design Decisions

### Hallucination Grounding
Before calling the LLM, scispaCy BC5CDR extracts every DISEASE and CHEMICAL entity from the document. The prompt explicitly instructs the model to only explain terms present in this list. After generation, the grounding checker verifies every explained term traces back to the source document. This achieved **100% grounding rate** across all evaluation documents.

### Document-Type-Specific Prompting
Four separate prompt templates handle each document type differently — a discharge summary needs medication warnings and follow-up flags, a lab report needs value interpretation, a radiology report needs imaging findings explained, and an insurance EOB needs billing terms clarified. Generic prompting produces significantly worse output quality.

### Pydantic Validation with Retry
LLM outputs are validated against strict Pydantic schemas. When validation fails (malformed JSON, missing fields, null values), a correction prompt is automatically sent to the LLM to fix the specific error. This prevents crashes from API inconsistency and maintains output reliability.

### Evaluation Harness
Three independent metrics measure output quality:
- **Flesch-Kincaid grade** — quantifies whether summaries are actually readable by patients
- **NER grounding rate** — quantifies whether the LLM is hallucinating terms not in the document
- **LLM-as-judge** — Groq Qwen 3.8B rates clarity, accuracy, and completeness independently

---

## Setup and Running Locally

### Prerequisites
- Python 3.11
- Node.js 18+
- Git

### Backend Setup

```bash
git clone https://github.com/Sahithi102302/medbridge.git
cd medbridge

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Install scispaCy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# Create .env file
cp .env.example .env
# Add your API keys to .env

# Start backend
uvicorn api.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Environment Variables

```
LLM_PROVIDER=google
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### Train the Classifier

The classifier requires the MIMIC-IV-Ext-BHC dataset (PhysioNet credentialed access required):

```bash
# After downloading mimic-iv-bhc.csv to data/samples/
jupyter notebook notebooks/01_eda_and_classifier_training.ipynb
```

### Run Evaluation

```bash
# Quick test (3 documents)
python run_eval.py --n 3

# Full evaluation (10 documents)
python run_eval.py --n 10

# View results dashboard
mlflow ui
# Open http://localhost:5000
```

---

## API Reference

### POST /analyze
Upload a PDF and receive full structured analysis.

**Request:** `multipart/form-data` with PDF file  
**Response:** JSON matching MedBridgeOutput schema

### POST /analyze/stream
Upload a PDF and receive live progress via Server-Sent Events.

**Events emitted:**
```
parsing → parsing_done
classifying → classifying_done
ner → ner_done
chunking → chunking_done
llm → llm_done
complete (with full result)
```

### GET /health
Returns server status.

```json
{"status": "ok", "version": "1.0.0", "service": "MedBridge API"}
```

---

## Output Schema

```json
{
  "summary": "Plain English summary, grade 5-7 reading level",
  "urgency_flags": [
    {
      "text": "Follow up with cardiologist within 48 hours",
      "timeframe": "within 48 hours",
      "severity": "critical"
    }
  ],
  "medications": [
    {
      "name": "clopidogrel",
      "brand_name": "Plavix",
      "purpose": "Prevents blood clots from forming in your new stent",
      "warning": "Do not stop without consulting your cardiologist",
      "frequency": "once daily, 75mg"
    }
  ],
  "jargon": [
    {
      "term": "NSTEMI",
      "explanation": "A type of heart attack where one artery is partially blocked",
      "confidence": 0.97,
      "source_sentence": "Final Diagnosis: Non-ST elevation myocardial infarction (NSTEMI)"
    }
  ],
  "questions": [
    "How long do I need to take both aspirin and clopidogrel?",
    "What symptoms should make me call 911 immediately?"
  ],
  "doc_type": "discharge",
  "overall_confidence": 0.95
}
```

---

## Dataset

**MIMIC-IV-Ext-BHC** (Medical Information Mart for Intensive Care)
- **Size:** 270,033 real de-identified hospital discharge notes
- **Source:** Beth Israel Deaconess Medical Center, 2008–2019
- **Published:** MIT, February 2025
- **Access:** PhysioNet credentialed access required
- **Citation:** Johnson, A. et al. (2025). MIMIC-IV-Ext-BHC. PhysioNet.

Access requires:
1. PhysioNet account with credentialed user status
2. Completion of CITI Data or Specimens Only Research training
3. Signing the PhysioNet Data Use Agreement

The dataset is excluded from this repository via `.gitignore` per PhysioNet terms.

---

## Classifier Training

The document classifier was trained on:
- **2,000** real MIMIC-IV discharge summaries (class 0)
- **1,000** synthetic lab reports (class 1)
- **1,000** synthetic radiology reports (class 2)
- **1,000** synthetic insurance EOBs (class 3)

**Features:** TF-IDF with 10,000 features, unigrams and bigrams, English stop words removed, sublinear TF scaling  
**Model:** Logistic Regression with balanced class weights  
**Results:** 100% accuracy on held-out test set, 80–96% confidence on real documents

See `notebooks/01_eda_and_classifier_training.ipynb` for full training details.

---

## What's Next — Phase 2

MedBridge Phase 2 extends the system with a **Prior Authorization RAG module** — automating insurance approval decisions by auditing patient records against policy criteria using ChromaDB vector search and an agentic LLM pipeline.

See the Phase 2 branch for implementation details.

---

## Author

**Sahithi Vankayala**  
M.S. Data Science, University of Maryland, College Park (2026)  
[GitHub](https://github.com/Sahithi102302) · [LinkedIn](https://linkedin.com/in/sahithivankayala)

---

## License

This project is for educational and portfolio purposes.  
MIMIC-IV data is subject to PhysioNet Data Use Agreement terms and is not included in this repository.
