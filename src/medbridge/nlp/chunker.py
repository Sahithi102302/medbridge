"""
chunker.py
----------
Step 4 of the MedBridge pipeline.

Job: Take clean text from parser.py and split it into
logical sections. Enforce a token budget so Gemini
never gets overwhelmed with too much text at once.

Why this exists:
- Medical documents can be very long (10-20+ pages)
- LLMs have token limits
- Different sections need different handling
  (medications section vs diagnosis section)
- Splitting by section gives Gemini focused context

Example input:
    "DISCHARGE SUMMARY
     Patient: Jane Doe
     
     ADMITTING DIAGNOSIS:
     Chest pain...
     
     DISCHARGE MEDICATIONS:
     1. Aspirin 81mg..."

Example output:
    [
        {"section": "HEADER",     "text": "DISCHARGE SUMMARY\nPatient: Jane Doe"},
        {"section": "DIAGNOSIS",  "text": "ADMITTING DIAGNOSIS:\nChest pain..."},
        {"section": "MEDICATIONS","text": "DISCHARGE MEDICATIONS:\n1. Aspirin 81mg..."},
    ]
"""

import re
import os
import sys
from typing import List, Dict


# maximum tokens per chunk before we split further
# gemini can handle more but this keeps responses focused
MAX_TOKENS_PER_CHUNK = 3000

# section headers commonly found in medical documents
# order matters — more specific patterns first
SECTION_PATTERNS = [
    # discharge summary sections
    r"DISCHARGE\s+DIAGNOSIS",
    r"ADMITTING\s+DIAGNOSIS",
    r"FINAL\s+DIAGNOSIS",
    r"DISCHARGE\s+MEDICATIONS?",
    r"MEDICATIONS?\s+ON\s+DISCHARGE",
    r"MEDICATIONS?\s+ON\s+ADMISSION",
    r"FOLLOW[\-\s]UP\s+INSTRUCTIONS?",
    r"DISCHARGE\s+INSTRUCTIONS?",
    r"PROCEDURE",
    r"HISTORY\s+OF\s+PRESENT\s+ILLNESS",
    r"ASSESSMENT\s+AND\s+PLAN",
    r"PHYSICAL\s+EXAMINATION",
    r"LABORATORY\s+(?:DATA|RESULTS?|VALUES?)",

    # lab report sections
    r"COMPREHENSIVE\s+METABOLIC",
    r"COMPLETE\s+BLOOD\s+COUNT",
    r"PROVIDER\s+NOTE",

    # radiology sections
    r"CLINICAL\s+INDICATION",
    r"FINDINGS?",
    r"IMPRESSION",
    r"RECOMMENDATION",
    r"TECHNIQUE",

    # insurance EOB sections
    r"SERVICES?\s+RENDERED",
    r"PAYMENT\s+SUMMARY",
    r"EXPLANATION\s+OF\s+BENEFITS",
]


def split_into_sections(text: str) -> List[Dict]:
    """
    Main function. Splits text into labeled sections.

    Args:
        text: clean text string from parser.py

    Returns:
        list of dicts, each with:
            section -> section name/label
            text    -> text content of that section
            tokens  -> approximate token count
    """
    # build one big regex pattern from all section patterns
    combined_pattern = "|".join(f"({p})" for p in SECTION_PATTERNS)

    # find all section headers and their positions
    matches = []
    for match in re.finditer(combined_pattern, text, re.IGNORECASE):
        matches.append((match.start(), match.group().strip()))

    # if no section headers found — treat whole document as one chunk
    if not matches:
        return [{
            "section": "FULL_DOCUMENT",
            "text": text,
            "tokens": estimate_tokens(text)
        }]

    # split text at each section header
    sections = []

    # text before first header = document header
    if matches[0][0] > 0:
        header_text = text[:matches[0][0]].strip()
        if header_text:
            sections.append({
                "section": "HEADER",
                "text": header_text,
                "tokens": estimate_tokens(header_text)
            })

    # each section from header to next header
    for i, (start_pos, section_name) in enumerate(matches):
        # end of this section = start of next section (or end of document)
        if i + 1 < len(matches):
            end_pos = matches[i + 1][0]
        else:
            end_pos = len(text)

        section_text = text[start_pos:end_pos].strip()

        if section_text:
            sections.append({
                "section": normalize_section_name(section_name),
                "text": section_text,
                "tokens": estimate_tokens(section_text)
            })

    # enforce token budget — split any oversized sections
    sections = enforce_token_budget(sections)

    return sections


def normalize_section_name(raw_name: str) -> str:
    """
    Converts raw section header text to a clean label.
    Example: "DISCHARGE MEDICATIONS" -> "MEDICATIONS"
    """
    raw_upper = raw_name.upper()

    if "DIAGNOSIS" in raw_upper:
        return "DIAGNOSIS"
    elif "MEDICATION" in raw_upper:
        return "MEDICATIONS"
    elif "FOLLOW" in raw_upper or "INSTRUCTION" in raw_upper:
        return "FOLLOW_UP"
    elif "PROCEDURE" in raw_upper:
        return "PROCEDURE"
    elif "FINDING" in raw_upper:
        return "FINDINGS"
    elif "IMPRESSION" in raw_upper:
        return "IMPRESSION"
    elif "INDICATION" in raw_upper:
        return "INDICATION"
    elif "RECOMMENDATION" in raw_upper:
        return "RECOMMENDATION"
    elif "LABORATORY" in raw_upper or "METABOLIC" in raw_upper:
        return "LAB_VALUES"
    elif "PROVIDER" in raw_upper:
        return "PROVIDER_NOTE"
    elif "SERVICES" in raw_upper:
        return "SERVICES"
    elif "PAYMENT" in raw_upper:
        return "PAYMENT"
    elif "HISTORY" in raw_upper:
        return "HISTORY"
    elif "ASSESSMENT" in raw_upper:
        return "ASSESSMENT"
    else:
        return raw_upper.replace(" ", "_")


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for a piece of text.
    Rule of thumb: 1 token ≈ 4 characters in English.
    This is an approximation — good enough for budget enforcement.
    """
    return len(text) // 4


def enforce_token_budget(sections: List[Dict]) -> List[Dict]:
    """
    Splits any section that exceeds MAX_TOKENS_PER_CHUNK
    into smaller pieces.
    """
    result = []
    for section in sections:
        if section["tokens"] <= MAX_TOKENS_PER_CHUNK:
            result.append(section)
        else:
            # split oversized section into paragraphs
            paragraphs = section["text"].split("\n\n")
            current_chunk = ""
            chunk_num = 1

            for para in paragraphs:
                if estimate_tokens(current_chunk + para) > MAX_TOKENS_PER_CHUNK:
                    if current_chunk:
                        result.append({
                            "section": f"{section['section']}_PART{chunk_num}",
                            "text": current_chunk.strip(),
                            "tokens": estimate_tokens(current_chunk)
                        })
                        chunk_num += 1
                        current_chunk = para
                else:
                    current_chunk += "\n\n" + para

            if current_chunk.strip():
                result.append({
                    "section": f"{section['section']}_PART{chunk_num}",
                    "text": current_chunk.strip(),
                    "tokens": estimate_tokens(current_chunk)
                })

    return result


def get_key_sections(sections: List[Dict]) -> List[Dict]:
    """
    Returns only the most important sections for translation.
    Skips header and payment sections which aren't medically useful.
    """
    skip_sections = {"HEADER", "PAYMENT", "SERVICES"}
    return [s for s in sections if s["section"] not in skip_sections]


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
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

        text = extract_text_from_pdf(pdf_path)
        sections = split_into_sections(text)

        print(f"Total sections found: {len(sections)}")
        print()
        for s in sections:
            print(f"  [{s['section']}] — {s['tokens']} tokens")
            print(f"  Preview: {s['text'][:80].strip()}")
            print()
        print("STATUS: OK")