"""
ner.py
------
Step 3 of the MedBridge pipeline.
NER is optional — if scispaCy is not installed, returns empty list.
"""

import os
import sys
from typing import List, Dict

# scispaCy is optional — not available in production deployment
try:
    import spacy
    nlp = spacy.load("en_ner_bc5cdr_md")
    NER_AVAILABLE = True
except Exception:
    NER_AVAILABLE = False
    nlp = None

# common false positives to ignore
IGNORE_TERMS = {
    "dob", "mrn", "ppo", "hmo", "eob", "npi",
    "date", "time", "patient", "name", "provider",
    "non-st", "non", "st"
}


def extract_medical_entities(text: str) -> List[Dict]:
    """
    Extracts medical entities from text.
    Returns empty list if scispaCy is not available.
    """
    if not NER_AVAILABLE or nlp is None:
        return []

    doc = nlp(text)
    entities = []
    seen_terms = set()

    for ent in doc.ents:
        term = ent.text.strip()
        if len(term) < 3:
            continue
        if term.replace(".", "").replace(",", "").isnumeric():
            continue
        if term.lower() in IGNORE_TERMS:
            continue
        if len(term.split()) <= 2 and "." in term:
            continue
        if term.lower() in seen_terms:
            continue
        seen_terms.add(term.lower())
        entities.append({
            "term": term,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return entities


def get_unique_terms(entities: List[Dict]) -> List[str]:
    return [e["term"] for e in entities]


def filter_by_label(entities: List[Dict], label: str) -> List[Dict]:
    return [e for e in entities if e["label"] == label]


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # add project root to path so we can import parser
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from medbridge.ingest.parser import extract_text_from_pdf

    sample_dir = "data/samples"
    pdfs = [
        "discharge_summary.pdf",
        "lab_report.pdf",
        "radiology_report.pdf",
        "insurance_eob.pdf",
    ]

    for pdf_name in pdfs:
        pdf_path = os.path.join(sample_dir, pdf_name)
        print(f"\n{'='*60}")
        print(f"FILE: {pdf_name}")
        print(f"{'='*60}")

        # step 1 - get clean text from parser
        text = extract_text_from_pdf(pdf_path)

        # step 2 - extract medical entities
        entities = extract_medical_entities(text)

        print(f"Total medical entities found: {len(entities)}")
        print(f"\nAll entities:")
        print("-" * 40)
        for e in entities:
            print(f"  {e['term']:<35} [{e['label']}]")
        print("-" * 40)
        print("STATUS: OK")