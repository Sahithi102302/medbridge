"""
grounding.py
------------
Checks that every jargon term Gemini explained actually
exists in the original document.

Why this matters:
- LLMs can hallucinate terms that were never in the document
- Grounding rate = grounded terms / total terms explained
- Target: above 90%
- This is your hallucination detection metric

How it works:
- Takes the NER entity list from ner.py (ground truth)
- Takes the jargon list from Gemini output
- For each explained term, checks if it (or a close match)
  appears in the original document text
- Returns grounding rate and flags ungrounded terms
"""

import re
from typing import List, Dict


def normalize(text: str) -> str:
    """
    Normalizes text for comparison.
    Lowercases, removes punctuation, strips whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_grounded(term: str, document_text: str) -> bool:
    """
    Checks if a term appears in the document text.
    Uses normalized comparison to handle case and punctuation.

    Args:
        term:          the jargon term Gemini explained
        document_text: the full original document text

    Returns:
        True if term is found in document, False if hallucinated
    """
    norm_term = normalize(term)
    norm_doc = normalize(document_text)

    # direct match
    if norm_term in norm_doc:
        return True

    # check if all words in the term appear in the document
    # handles cases like "left anterior descending artery"
    # where the document might say "LAD (left anterior descending)"
    term_words = norm_term.split()
    if len(term_words) > 1:
        # at least 80% of words must appear in document
        words_found = sum(1 for w in term_words if w in norm_doc and len(w) > 2)
        if words_found / len(term_words) >= 0.8:
            return True

    # check abbreviation — if term is an acronym, check if it appears
    if len(term) <= 6 and term.isupper():
        if term.lower() in norm_doc:
            return True

    return False


def evaluate_grounding(
    jargon_items: List[Dict],
    document_text: str,
    entities: List[Dict] = None
) -> Dict:
    """
    Main function. Evaluates grounding rate for one document analysis.

    Args:
        jargon_items:  list of jargon dicts from MedBridgeOutput
        document_text: original document text from parser.py
        entities:      NER entities from ner.py (optional extra check)

    Returns:
        dict with grounding rate, grounded terms, ungrounded terms
    """
    if not jargon_items:
        return {
            "grounding_rate": 1.0,
            "grounded_count": 0,
            "ungrounded_count": 0,
            "total_count": 0,
            "grounded_terms": [],
            "ungrounded_terms": [],
            "passed": True,
        }

    grounded = []
    ungrounded = []

    for item in jargon_items:
        term = item.get("term", "")
        source_sentence = item.get("source_sentence", "")

        # check 1 — term in document
        term_grounded = is_grounded(term, document_text)

        # check 2 — source sentence in document (if provided)
        source_grounded = True
        if source_sentence:
            source_grounded = is_grounded(source_sentence[:50], document_text)

        if term_grounded:
            grounded.append({
                "term": term,
                "source_verified": source_grounded
            })
        else:
            ungrounded.append({
                "term": term,
                "reason": "Term not found in document text"
            })

    total = len(jargon_items)
    grounding_rate = len(grounded) / total if total > 0 else 1.0
    passed = grounding_rate >= 0.90

    return {
        "grounding_rate": round(grounding_rate, 3),
        "grounding_rate_pct": f"{grounding_rate * 100:.1f}%",
        "grounded_count": len(grounded),
        "ungrounded_count": len(ungrounded),
        "total_count": total,
        "grounded_terms": grounded,
        "ungrounded_terms": ungrounded,
        "passed": passed,
        "target": "90% grounding rate",
    }


def evaluate_batch_grounding(results: List[Dict]) -> Dict:
    """
    Aggregates grounding results across multiple documents.
    Used by run_eval.py for the 50-doc evaluation.

    Args:
        results: list of grounding result dicts

    Returns:
        aggregate stats dict
    """
    if not results:
        return {}

    rates = [r["grounding_rate"] for r in results]
    passed = [r for r in results if r["passed"]]

    return {
        "count": len(results),
        "avg_grounding_rate": round(sum(rates) / len(rates), 3),
        "avg_grounding_rate_pct": f"{sum(rates) / len(rates) * 100:.1f}%",
        "min_grounding_rate": min(rates),
        "max_grounding_rate": max(rates),
        "pass_rate": round(len(passed) / len(results) * 100, 1),
    }


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from medbridge.ingest.parser import extract_text_from_pdf

    sample_pdf = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "data", "samples", "discharge_summary.pdf"
    )

    print("=== Grounding Evaluation Test ===\n")

    text = extract_text_from_pdf(sample_pdf)

    # simulate jargon output — mix of grounded and ungrounded
    test_jargon = [
        {
            "term": "NSTEMI",
            "source_sentence": "Final diagnosis NSTEMI"
        },
        {
            "term": "Percutaneous coronary intervention",
            "source_sentence": "Percutaneous coronary intervention (PCI)"
        },
        {
            "term": "Drug-eluting stent",
            "source_sentence": "Drug-eluting stent placed in left anterior descending artery"
        },
        {
            "term": "Troponin",
            "source_sentence": "Troponin peak was 4.2 ng per mL on admission"
        },
        {
            "term": "Quantum entanglement therapy",  # hallucinated term
            "source_sentence": "Patient underwent quantum entanglement therapy"
        },
        {
            "term": "Nanobots",  # another hallucinated term
            "source_sentence": "Nanobots were deployed"
        },
    ]

    result = evaluate_grounding(test_jargon, text)

    print(f"Total terms:      {result['total_count']}")
    print(f"Grounded:         {result['grounded_count']}")
    print(f"Ungrounded:       {result['ungrounded_count']}")
    print(f"Grounding rate:   {result['grounding_rate_pct']}")
    print(f"Passed (>=90%):   {result['passed']}")
    print(f"\nGrounded terms:")
    for t in result["grounded_terms"]:
        print(f"  ✓ {t['term']}")
    print(f"\nUngrounded terms (potential hallucinations):")
    for t in result["ungrounded_terms"]:
        print(f"  ✗ {t['term']} — {t['reason']}")