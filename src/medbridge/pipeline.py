"""
pipeline.py
-----------
The main orchestrator of MedBridge.

Job: Take a PDF file path, run it through every step,
return a complete MedBridgeOutput object.

This is the single function the FastAPI backend calls.
It connects all 5 modules:
    1. parser.py     → extract text from PDF
    2. classifier.py → detect document type
    3. ner.py        → extract medical entities
    4. chunker.py    → split into sections
    5. client.py     → call Gemini, return structured output

Usage:
    from medbridge.pipeline import run_pipeline
    result = run_pipeline("path/to/document.pdf")
    print(result.summary)
"""

import os
import sys
import time
from typing import Optional, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medbridge.ingest.parser import extract_text_from_pdf, get_page_count
from medbridge.nlp.classifier import classify_document
from medbridge.nlp.ner import extract_medical_entities
from medbridge.nlp.chunker import split_into_sections
from medbridge.llm.client import analyze_document
from medbridge.llm.schemas import MedBridgeOutput


def run_pipeline(
    pdf_path: str,
    progress_callback: Optional[Callable] = None
) -> MedBridgeOutput:
    """
    Main pipeline function. Takes a PDF path, returns MedBridgeOutput.

    Args:
        pdf_path:          path to the PDF file
        progress_callback: optional function called after each step
                          used by FastAPI SSE to stream progress to frontend
                          signature: callback(step: str, detail: str)

    Returns:
        MedBridgeOutput — complete validated analysis result
    """

    def progress(step: str, detail: str = ""):
        """Helper to emit progress if callback provided."""
        if progress_callback:
            progress_callback(step, detail)
        print(f"  [{step}] {detail}")

    start_time = time.time()

    # ── STEP 1: Parse PDF ────────────────────────────────────
    progress("parsing", "Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    pages = get_page_count(pdf_path)
    word_count = len(text.split())
    progress("parsing_done", f"{pages} pages, {word_count} words extracted")

    # ── STEP 2: Classify document ────────────────────────────
    progress("classifying", "Detecting document type...")
    doc_type, classifier_confidence = classify_document(text)
    progress("classifying_done", f"Detected: {doc_type} ({classifier_confidence:.0%} confidence)")

    # ── STEP 3: Extract medical entities ────────────────────
    progress("ner", "Extracting medical entities...")
    entities = extract_medical_entities(text)
    diseases = [e for e in entities if e["label"] == "DISEASE"]
    chemicals = [e for e in entities if e["label"] == "CHEMICAL"]
    progress("ner_done", f"{len(entities)} entities found ({len(diseases)} diseases, {len(chemicals)} medications)")

    # ── STEP 4: Split into sections ──────────────────────────
    progress("chunking", "Splitting document into sections...")
    sections = split_into_sections(text)
    progress("chunking_done", f"{len(sections)} sections identified")

    # ── STEP 5: Call Gemini ──────────────────────────────────
    progress("llm", "Translating with Gemini...")
    result = analyze_document(text, entities, sections, doc_type)
    progress("llm_done", f"Translation complete ({result.overall_confidence:.0%} confidence)")

    # ── DONE ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    progress("complete", f"Finished in {elapsed:.1f} seconds")

    return result


def run_pipeline_from_bytes(
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    progress_callback: Optional[Callable] = None
) -> MedBridgeOutput:
    """
    Alternative entry point — accepts PDF as bytes instead of file path.
    Used by FastAPI which receives uploaded files as bytes.

    Args:
        pdf_bytes:         raw PDF file bytes from upload
        filename:          original filename for logging
        progress_callback: optional progress callback

    Returns:
        MedBridgeOutput
    """
    import tempfile

    # write bytes to a temporary file
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False,
        prefix="medbridge_"
    ) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path, progress_callback)
    finally:
        # always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return result


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "samples"
    )

    pdfs = [
        "discharge_summary.pdf",
        "lab_report.pdf",
        "radiology_report.pdf",
        "insurance_eob.pdf",
    ]

    for pdf_name in pdfs:
        pdf_path = os.path.join(sample_dir, pdf_name)
        print(f"\n{'='*60}")
        print(f"PIPELINE TEST: {pdf_name}")
        print(f"{'='*60}")

        result = run_pipeline(pdf_path)

        print(f"\nDOC TYPE:   {result.doc_type}")
        print(f"CONFIDENCE: {result.overall_confidence:.0%}")
        print(f"\nSUMMARY:")
        print(f"  {result.summary}")
        print(f"\nURGENCY FLAGS: {len(result.urgency_flags)}")
        for flag in result.urgency_flags:
            print(f"  [{flag.severity.upper()}] {flag.text}")
        print(f"\nMEDICATIONS: {len(result.medications)}")
        for med in result.medications:
            print(f"  {med.name} — {med.purpose[:50]}...")
        print(f"\nJARGON TERMS: {len(result.jargon)}")
        for j in result.jargon:
            print(f"  {j.term} ({j.confidence:.0%})")
        print(f"\nDOCTOR QUESTIONS: {len(result.questions)}")
        for i, q in enumerate(result.questions, 1):
            print(f"  {i}. {q}")
        print(f"\nSTATUS: OK")
        print()