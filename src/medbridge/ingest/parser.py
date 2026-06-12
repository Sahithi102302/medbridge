"""
parser.py
---------
Step 1 of the MedBridge pipeline.

Job: Take a PDF file path, extract all text cleanly, return a single string.

Why this exists:
- PDFs are not plain text files
- Medical PDFs have noisy headers, footers, special characters
- Everything downstream (classifier, NER, Gemini) needs clean text
"""

import fitz  # PyMuPDF
import re
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Main function. Takes a PDF file path, returns clean extracted text.

    Args:
        pdf_path: full path to the PDF file

    Returns:
        clean text string ready for downstream processing
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    all_pages_text = []
    for page in doc:
        blocks = page.get_text("blocks")
        page_text = ""
        for block in blocks:
            if block[6] == 0:  # text block only, skip images
                page_text += block[4] + "\n"
        all_pages_text.append(page_text)

    doc.close()

    cleaned_pages = remove_headers_footers(all_pages_text)
    full_text = "\n".join(cleaned_pages)
    full_text = clean_text(full_text)

    return full_text


def remove_headers_footers(pages: list) -> list:
    """
    Detects lines that appear on multiple pages and removes them.
    These are usually patient name, MRN, page numbers repeated
    on every page of a hospital document.
    """
    if len(pages) <= 1:
        return pages

    line_counts = {}
    for page_text in pages:
        lines = page_text.split("\n")
        edge_lines = lines[:3] + lines[-3:]
        for line in edge_lines:
            line = line.strip()
            if len(line) > 5:
                line_counts[line] = line_counts.get(line, 0) + 1

    threshold = max(2, len(pages) // 2)
    repeated_lines = {
        line for line, count in line_counts.items()
        if count >= threshold
    }

    cleaned = []
    for page_text in pages:
        lines = page_text.split("\n")
        filtered = [
            line for line in lines
            if line.strip() not in repeated_lines
        ]
        cleaned.append("\n".join(filtered))

    return cleaned


def clean_text(text: str) -> str:
    """
    Cleans up extracted text.
    - Removes page break characters
    - Fixes garbled medical symbols
    - Removes excessive blank lines
    - Strips trailing whitespace
    """
    text = text.replace("\f", "\n")

    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b0": " degrees ",
        "\u00b5": "u",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00b1": "+/-",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def get_page_count(pdf_path: str) -> int:
    """Returns number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
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

        try:
            text = extract_text_from_pdf(pdf_path)
            pages = get_page_count(pdf_path)
            print(f"Pages: {pages}")
            print(f"Characters extracted: {len(text)}")
            print(f"Words extracted: {len(text.split())}")
            print(f"\nFirst 400 characters:")
            print("-" * 40)
            print(text[:400])
            print("-" * 40)
            print("STATUS: OK")
        except Exception as e:
            print(f"ERROR: {e}")