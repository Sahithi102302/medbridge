"""
readability.py
--------------
Measures Flesch-Kincaid readability grade of generated summaries.

Why this matters:
- MedBridge aims for grade 6-8 reading level
- Grade 6 = sixth grader can understand it
- If a summary scores above 8, it's too complex for most patients
- This is a quantifiable quality metric for your resume and README

Flesch-Kincaid Grade Formula:
FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
Lower score = easier to read
"""

import textstat
from typing import Dict, List


TARGET_MAX_GRADE = 8.0
TARGET_MIN_GRADE = 2.0


def compute_fk_grade(text: str) -> float:
    """
    Computes Flesch-Kincaid grade level for a piece of text.

    Args:
        text: plain English text to evaluate

    Returns:
        float — grade level (e.g. 6.2 means grade 6 reading level)
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    return round(textstat.flesch_kincaid_grade(text), 2)


def compute_reading_ease(text: str) -> float:
    """
    Computes Flesch Reading Ease score.
    Higher = easier to read (0-100 scale)
    90-100 = very easy, 0-30 = very difficult

    Args:
        text: plain English text

    Returns:
        float — reading ease score
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    return round(textstat.flesch_reading_ease(text), 2)


def evaluate_summary(summary: str) -> Dict:
    """
    Full readability evaluation of one summary.

    Args:
        summary: the plain English summary from MedBridge

    Returns:
        dict with grade, ease, pass/fail, and feedback
    """
    fk_grade = compute_fk_grade(summary)
    reading_ease = compute_reading_ease(summary)
    passed = TARGET_MIN_GRADE <= fk_grade <= TARGET_MAX_GRADE

    if fk_grade < TARGET_MIN_GRADE:
        feedback = "Summary may be too simple — check for missing context"
    elif fk_grade <= TARGET_MAX_GRADE:
        feedback = "Readability is within target range"
    elif fk_grade <= 10:
        feedback = "Slightly above target — consider simplifying sentence structure"
    else:
        feedback = "Too complex for most patients — needs significant simplification"

    return {
        "fk_grade": fk_grade,
        "reading_ease": reading_ease,
        "passed": passed,
        "target": f"Grade {TARGET_MIN_GRADE} to {TARGET_MAX_GRADE}",
        "feedback": feedback,
        "word_count": len(summary.split()),
        "sentence_count": textstat.sentence_count(summary),
    }


def evaluate_batch(summaries: List[str]) -> Dict:
    """
    Evaluates readability across multiple summaries.
    Used by the eval harness to score 50 gold documents.

    Args:
        summaries: list of plain English summaries

    Returns:
        dict with aggregate stats
    """
    if not summaries:
        return {}

    results = [evaluate_summary(s) for s in summaries]
    grades = [r["fk_grade"] for r in results]
    passed = [r for r in results if r["passed"]]

    return {
        "count": len(summaries),
        "avg_fk_grade": round(sum(grades) / len(grades), 2),
        "min_fk_grade": min(grades),
        "max_fk_grade": max(grades),
        "pass_rate": round(len(passed) / len(summaries) * 100, 1),
        "individual_results": results,
    }


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_summaries = [
        # good — grade 6-8
        "You came to the hospital with chest pain and were found to have "
        "a type of heart attack. Doctors placed a small tube in your artery "
        "to keep it open. You need to take two medications every day.",

        # too complex — grade 10+
        "The patient presented with acute coronary syndrome and subsequently "
        "underwent percutaneous coronary intervention with drug-eluting stent "
        "placement in the left anterior descending artery.",

        # too simple — grade 3
        "You had a heart attack. You got a stent. Take your pills.",
    ]

    labels = ["Good (target)", "Too complex", "Too simple"]

    print("=== Readability Evaluation Test ===\n")
    for label, summary in zip(labels, test_summaries):
        result = evaluate_summary(summary)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{label}]")
        print(f"  FK Grade:      {result['fk_grade']} — {status}")
        print(f"  Reading ease:  {result['reading_ease']}")
        print(f"  Words:         {result['word_count']}")
        print(f"  Sentences:     {result['sentence_count']}")
        print(f"  Feedback:      {result['feedback']}")
        print()

    print("=== Batch evaluation ===")
    batch = evaluate_batch(test_summaries)
    print(f"  Count:         {batch['count']}")
    print(f"  Avg FK grade:  {batch['avg_fk_grade']}")
    print(f"  Pass rate:     {batch['pass_rate']}%")
    print(f"  Min grade:     {batch['min_fk_grade']}")
    print(f"  Max grade:     {batch['max_fk_grade']}")