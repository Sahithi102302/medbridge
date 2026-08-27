"""
judge.py
--------
LLM-as-judge evaluation — uses Gemini to rate MedBridge output
on clarity, accuracy, and completeness.

Why this matters:
- Human evaluation is expensive and slow
- LLM-as-judge is a standard research technique
- Gives a 1-5 score on three dimensions
- Tracked in MLflow for experiment comparison
"""

import os
import json
import re
import sys
from typing import Dict
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

JUDGE_MODEL = "gemini-2.5-flash"

JUDGE_PROMPT = """You are a medical communication expert evaluating AI-generated patient summaries.

Rate the following medical document translation on THREE dimensions.
Be strict and objective. Do not give high scores unless truly deserved.

SCORING RUBRIC:

CLARITY (1-5):
5 = Perfect plain English, any adult can understand, no jargon unexplained
4 = Mostly clear, minor complex phrases
3 = Acceptable but some medical terms left unexplained
2 = Difficult for average patient to understand
1 = Still full of medical jargon

ACCURACY (1-5):
5 = Everything matches the original document perfectly
4 = Minor omissions but nothing misleading
3 = Some important information missing
2 = Contains incorrect information
1 = Significantly wrong or misleading

COMPLETENESS (1-5):
5 = All important information captured
4 = Most important information captured
3 = Key information present but some gaps
2 = Significant gaps in important information
1 = Missing most important information

You will be given the original document and the MedBridge output.
Return ONLY this JSON — no text before or after, no markdown:
{
    "clarity": 4,
    "accuracy": 5,
    "completeness": 4,
    "overall": 4.3,
    "feedback": "One sentence of specific feedback"
}"""


def judge_output(
    original_text: str,
    summary: str,
    medications: list,
    jargon: list,
    questions: list
) -> Dict:
    """
    Uses Gemini to evaluate MedBridge output quality.
    """
    med_text = "\n".join([
        f"- {m.get('name', '')} ({m.get('frequency', '')}): {m.get('purpose', '')}"
        for m in medications
    ]) if medications else "None listed"

    jargon_text = "\n".join([
        f"- {j.get('term', '')}: {j.get('explanation', '')}"
        for j in jargon
    ]) if jargon else "None listed"

    questions_text = "\n".join([
        f"{i+1}. {q}" for i, q in enumerate(questions)
    ]) if questions else "None listed"

    user_message = f"""ORIGINAL DOCUMENT:
{original_text[:2000]}

MEDBRIDGE OUTPUT TO EVALUATE:

SUMMARY:
{summary}

MEDICATIONS:
{med_text}

JARGON EXPLAINED:
{jargon_text}

QUESTIONS FOR DOCTOR:
{questions_text}

Rate this output. Return ONLY this exact JSON structure with no extra text:
{{"clarity": X, "accuracy": X, "completeness": X, "overall": X, "feedback": "one sentence"}}"""
    try:
        model = genai.GenerativeModel(
            model_name=JUDGE_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            )
        )

        # combine system prompt and user message into one
        full_prompt = JUDGE_PROMPT + "\n\n" + user_message

        response = model.generate_content(full_prompt)
        raw = response.text.strip()
        #print(f"DEBUG raw response:\n{raw}\n")

        if not raw:
            print("  Warning: empty response from judge")
            return {
                "clarity": 3,
                "accuracy": 3,
                "completeness": 3,
                "overall": 3.0,
                "feedback": "Judge returned empty response",
                "passed": False
            }

        # strip markdown code blocks if present
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        raw = raw.strip()

        # extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            scores = json.loads(json_match.group(0))
        else:
            scores = json.loads(raw)

        # compute overall as average of three scores
        scores["overall"] = round(
            (scores["clarity"] + scores["accuracy"] + scores["completeness"]) / 3, 2
        )
        scores["passed"] = scores["overall"] >= 4.0

        return scores

    except Exception as e:
        print(f"  Judge error: {e}")
        return {
            "clarity": 0,
            "accuracy": 0,
            "completeness": 0,
            "overall": 0,
            "feedback": f"Evaluation failed: {str(e)}",
            "passed": False
        }


def evaluate_batch_judge(judge_results: list) -> Dict:
    """
    Aggregates judge scores across multiple documents.
    """
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


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from medbridge.ingest.parser import extract_text_from_pdf

    sample_pdf = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "data", "samples", "discharge_summary.pdf"
    )

    print("=== LLM-as-Judge Evaluation Test ===\n")
    print("Loading sample document...")
    text = extract_text_from_pdf(sample_pdf)

    test_summary = (
        "You were admitted to the hospital for chest pain and diagnosed "
        "with a type of heart attack called a Non-ST elevation myocardial "
        "infarction (NSTEMI). Doctors performed a procedure to place a stent "
        "in one of your heart arteries to help blood flow. You have new "
        "medications and important follow-up appointments to help with your recovery."
    )

    test_medications = [
        {"name": "Aspirin", "frequency": "81mg once daily",
         "purpose": "Helps prevent blood clots after your heart attack and stent placement"},
        {"name": "Clopidogrel", "frequency": "75mg once daily",
         "purpose": "Helps prevent blood clots after your heart attack and stent placement"},
        {"name": "Atorvastatin", "frequency": "40mg every evening",
         "purpose": "Helps lower your cholesterol and prevent future heart problems"},
    ]

    test_jargon = [
        {"term": "NSTEMI",
         "explanation": "A type of heart attack where blood flow to the heart is partially blocked"},
        {"term": "PCI",
         "explanation": "A procedure to open a blocked heart artery using a balloon and stent"},
        {"term": "Drug-eluting stent",
         "explanation": "A small mesh tube that releases medicine to keep your artery open"},
    ]

    test_questions = [
        "What activities should I avoid after getting a stent?",
        "What are the side effects of my new medications?",
        "When can I return to work or normal activities?",
    ]

    print("Calling Gemini judge...\n")
    result = judge_output(
        text, test_summary, test_medications, test_jargon, test_questions
    )

    print(f"Clarity:        {result['clarity']}/5")
    print(f"Accuracy:       {result['accuracy']}/5")
    print(f"Completeness:   {result['completeness']}/5")
    print(f"Overall:        {result['overall']}/5")
    print(f"Passed (>=4.0): {result['passed']}")
    print(f"Feedback:       {result.get('feedback', 'N/A')}")