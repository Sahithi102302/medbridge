"""
classifier.py
-------------
Step 2 of the MedBridge pipeline.

Job: Take clean text from parser.py and classify it into
one of four document types:
    0 = discharge summary
    1 = lab report
    2 = radiology report
    3 = insurance EOB

Why this exists:
- Each document type needs a different prompt template
- A discharge summary needs different questions than a lab report
- Knowing the type upfront makes Gemini output much more accurate

How it works:
- TF-IDF converts text to numerical features
- Logistic Regression predicts the document type
- Both models were trained in notebooks/01_eda_and_classifier_training.ipynb
- Models are loaded from data/models/ folder
"""

import joblib
import os
from typing import Tuple

# label mapping
LABEL_NAMES = {
    0: "discharge",
    1: "lab",
    2: "radiology",
    3: "eob"
}

# paths to saved models
MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "models"
)
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "doc_classifier.joblib")


def load_models():
    """
    Loads the trained vectorizer and classifier from disk.
    Raises a clear error if models haven't been trained yet.
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at {VECTORIZER_PATH}\n"
            "Run the notebook: notebooks/01_eda_and_classifier_training.ipynb"
        )
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}\n"
            "Run the notebook: notebooks/01_eda_and_classifier_training.ipynb"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    return vectorizer, classifier


# load models once when module is imported
vectorizer, classifier = load_models()


def classify_document(text: str) -> Tuple[str, float]:
    """
    Main function. Takes clean text, returns document type and confidence.

    Args:
        text: clean text string from parser.py

    Returns:
        tuple of (doc_type, confidence)
        doc_type   -> "discharge", "lab", "radiology", or "eob"
        confidence -> float between 0 and 1
    """
    # convert text to TF-IDF features
    features = vectorizer.transform([text])

    # predict class
    pred = classifier.predict(features)[0]

    # get confidence (probability of predicted class)
    probs = classifier.predict_proba(features)[0]
    confidence = float(probs[pred])

    doc_type = LABEL_NAMES[pred]

    return doc_type, confidence


def classify_with_all_probs(text: str) -> dict:
    """
    Returns probabilities for all 4 classes.
    Useful for debugging and the debug tab in the frontend.

    Args:
        text: clean text string

    Returns:
        dict with doc type as key and probability as value
    """
    features = vectorizer.transform([text])
    probs = classifier.predict_proba(features)[0]

    return {
        LABEL_NAMES[i]: float(probs[i])
        for i in range(len(LABEL_NAMES))
    }


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from medbridge.ingest.parser import extract_text_from_pdf

    sample_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "samples"
    )

    pdfs = [
        ("discharge_summary.pdf", "discharge"),
        ("lab_report.pdf",        "lab"),
        ("radiology_report.pdf",  "radiology"),
        ("insurance_eob.pdf",     "eob"),
    ]

    print("=== Testing classifier.py ===\n")
    for pdf_name, true_label in pdfs:
        pdf_path = os.path.join(sample_dir, pdf_name)
        text = extract_text_from_pdf(pdf_path)

        doc_type, confidence = classify_document(text)
        all_probs = classify_with_all_probs(text)

        status = "CORRECT" if doc_type == true_label else "WRONG"

        print(f"File:      {pdf_name}")
        print(f"True:      {true_label}")
        print(f"Predicted: {doc_type} ({confidence:.2%})")
        print(f"All probs: { {k: f'{v:.2%}' for k,v in all_probs.items()} }")
        print(f"Status:    {status}")
        print()