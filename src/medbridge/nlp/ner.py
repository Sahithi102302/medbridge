"""
ner.py
------
Step 3 of the MedBridge pipeline.

Job: Take clean text from parser.py, find all medical entities in it.
Returns a list of medical terms with their labels.

Why this exists:
- Tells Gemini exactly WHICH terms to explain
- Used to verify Gemini only explains terms that actually
  exist in the document (hallucination grounding)

Example:
    Input:  "Patient diagnosed with NSTEMI. Started on
             clopidogrel 75mg and atorvastatin 40mg."

    Output: [
        {"term": "NSTEMI",       "label": "DISEASE"},
        {"term": "clopidogrel",  "label": "CHEMICAL"},
        {"term": "atorvastatin", "label": "CHEMICAL"},
    ]
"""

import spacy
import os
import sys
from typing import List, Dict

# load scispaCy medical model once when module is imported
# loading once is important — loading every call is very slow
nlp = spacy.load("en_ner_bc5cdr_md")


# common false positives to ignore
IGNORE_TERMS = {
    "dob", "mrn", "ppo", "hmo", "eob", "npi",
    "date", "time", "patient", "name", "provider",
    "non-st", "non", "st"
}

def extract_medical_entities(text: str) -> List[Dict]:
    """
    Main function. Takes clean text, returns list of medical entities.

    Args:
        text: clean text string from parser.py

    Returns:
        list of dicts, each with:
            term  -> the medical term found
            label -> DISEASE or CHEMICAL
            start -> character position where it starts
            end   -> character position where it ends
    """
    doc = nlp(text)

    entities = []
    seen_terms = set()

    for ent in doc.ents:
        term = ent.text.strip()

        # skip if too short
        if len(term) < 3:
            continue

        # skip pure numbers
        if term.replace(".", "").replace(",", "").isnumeric():
            continue

        # skip known false positives
        if term.lower() in IGNORE_TERMS:
            continue

        # skip if looks like a person's name
        # (single letter followed by period and word)
        if len(term.split()) <= 2 and "." in term:
            continue

        # skip duplicates
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
    """
    Returns just the term strings from entity list.
    Used by hallucination grounding checker later.
    """
    return [e["term"] for e in entities]


def filter_by_label(entities: List[Dict], label: str) -> List[Dict]:
    """
    Returns only entities with a specific label.
    Example: filter_by_label(entities, "DISEASE")
    """
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