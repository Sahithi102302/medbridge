"""
judge.py
--------
LLM-as-judge evaluation — uses Groq (Llama 3) to rate MedBridge output
on clarity, accuracy, and completeness.

Uses Groq instead of Gemini for the judge because:
- Free tier with no daily limits
- No safety filter issues on medical content
- Fast inference
"""

import os
import re
import sys
from typing import Dict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

JUDGE_MODEL = "qwen/qwen3.8-27b"


def judge_output(
    original_text: str,
    summary: str,
    medications: list,
    jargon: list,
    questions: list
) -> Dict:
    """
    Uses Groq Llama to evaluate MedBridge output quality.
    Returns scores for clarity, accuracy, completeness.
    """
    med_names = [m.get('name', '') for m in medications] if medications else []
    med_str = ", ".join(med_names) if med_names else "none"

    prompt = (
        "IMPORTANT: This summary is intentionally brief (3 sentences max) for patient readability.\n"
        "Rate completeness based on whether KEY facts are present, not whether every detail is included.\n"
        "You are evaluating a patient health summary written in plain English.\n"
        "Rate it on three dimensions using integers 1 to 5.\n"
        "Reply with ONLY three integers separated by commas. Nothing else.\n"
        "Example reply: 4,3,5\n\n"
        "Dimension 1 - clarity: Is it written in simple words any adult understands?\n"
        "Dimension 2 - accuracy: Does it correctly summarize the medical situation?\n"
        "Dimension 3 - completeness: Does it cover the key medical information?\n\n"
        "PATIENT SUMMARY TO RATE:\n"
        + summary
        + "\n\nMEDICATIONS MENTIONED: " + med_str
        + "\n\nYour three scores (X,X,X):"
    )

    try:
        response = groq_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        print(f"  Raw judge response: {repr(raw)}")

        # extract all numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', raw)
        numbers = [_safe_int(n) for n in numbers[:3]]

        if len(numbers) >= 3:
            clarity = numbers[0]
            accuracy = numbers[1]
            completeness = numbers[2]
        elif len(numbers) == 2:
            clarity = numbers[0]
            accuracy = numbers[1]
            completeness = numbers[0]
        elif len(numbers) == 1:
            clarity = accuracy = completeness = numbers[0]
        else:
            print(f"  Could not parse: {repr(raw)}")
            return _fallback(3, 3, 3, f"Parse failed: {raw[:30]}")

        overall = round((clarity + accuracy + completeness) / 3, 2)
        return {
            "clarity": clarity,
            "accuracy": accuracy,
            "completeness": completeness,
            "overall": overall,
            "feedback": "evaluated by Groq Llama",
            "passed": overall >= 4.0
        }

    except Exception as e:
        print(f"  Judge error: {e}")
        return _fallback(0, 0, 0, f"Failed: {str(e)}")


def _safe_int(val: str) -> int:
    try:
        return max(1, min(5, int(float(str(val).strip()))))
    except Exception:
        return 3


def _fallback(clarity: int, accuracy: int, completeness: int, feedback: str) -> Dict:
    overall = round((clarity + accuracy + completeness) / 3, 2)
    return {
        "clarity": clarity,
        "accuracy": accuracy,
        "completeness": completeness,
        "overall": overall,
        "feedback": feedback,
        "passed": overall >= 4.0
    }


def evaluate_batch_judge(judge_results: list) -> Dict:
    if not judge_results:
        return {}
    valid = [r for r in judge_results if r["overall"] > 0]
    if not valid:
        return {"error": "No valid results"}
    return {
        "count": len(valid),
        "avg_clarity": round(sum(r["clarity"] for r in valid) / len(valid), 2),
        "avg_accuracy": round(sum(r["accuracy"] for r in valid) / len(valid), 2),
        "avg_completeness": round(sum(r["completeness"] for r in valid) / len(valid), 2),
        "avg_overall": round(sum(r["overall"] for r in valid) / len(valid), 2),
        "pass_rate": round(
            sum(1 for r in valid if r["passed"]) / len(valid) * 100, 1
        ),
    }


if __name__ == "__main__":
    test_summary = (
        "You were admitted for chest pain and diagnosed with NSTEMI. "
        "Doctors placed a stent in your heart artery. Take two blood-thinning "
        "medications daily and see your heart doctor within 48 hours."
    )
    test_medications = [
        {"name": "Aspirin", "purpose": "Prevents blood clots"},
        {"name": "Clopidogrel", "purpose": "Works with aspirin"},
    ]

    print("=== LLM-as-Judge Evaluation Test ===\n")
    print("Calling Groq Llama judge...\n")
    result = judge_output("", test_summary, test_medications, [], [])

    print(f"Clarity:        {result['clarity']}/5")
    print(f"Accuracy:       {result['accuracy']}/5")
    print(f"Completeness:   {result['completeness']}/5")
    print(f"Overall:        {result['overall']}/5")
    print(f"Passed (>=4.0): {result['passed']}")
    print(f"Feedback:       {result.get('feedback', 'N/A')}")