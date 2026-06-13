"""
schemas.py
----------
Defines the exact structure of MedBridge output.

Why this exists:
- Gemini is an LLM — it can return anything
- We need consistent, predictable output every time
- Pydantic validates the output and raises clear errors
  if anything is missing or wrong
- Every other part of the system (API, frontend) relies
  on this structure being consistent

Output structure:
{
    "summary": "plain English summary...",
    "urgency_flags": [
        {"text": "Follow up in 48 hours", "timeframe": "48 hours", "severity": "critical"}
    ],
    "medications": [
        {"name": "Aspirin", "brand_name": "Bayer", "purpose": "prevents clots",
         "warning": "do not stop", "frequency": "once daily"}
    ],
    "jargon": [
        {"term": "NSTEMI", "explanation": "a type of heart attack...",
         "confidence": 0.95, "source_sentence": "Patient had NSTEMI..."}
    ],
    "questions": ["What does this mean for my daily life?", ...],
    "doc_type": "discharge",
    "overall_confidence": 0.87
}
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class UrgencyFlag(BaseModel):
    """
    Something the patient needs to act on.
    Examples:
    - "Follow up with cardiologist within 48 hours"
    - "Do not stop clopidogrel"
    - "Call 911 if chest pain returns"
    """
    text: str = Field(description="Plain English description of what to do")
    timeframe: str = Field(description="When to do it e.g. '48 hours', 'immediately', 'ongoing'")
    severity: Literal["critical", "soon", "routine"] = Field(
        description="critical=within 48hrs, soon=within weeks, routine=ongoing"
    )


class MedicationItem(BaseModel):
    """
    One medication from the discharge papers.
    """
    name: str = Field(description="Generic drug name e.g. clopidogrel")
    brand_name: Optional[str] = Field(
        default=None,
        description="Brand name if known e.g. Plavix"
    )
    purpose: str = Field(
        description="What this drug does in plain English, one sentence"
    )
    warning: Optional[str] = Field(
        default=None,
        description="Critical warning if any e.g. do not stop without consulting doctor"
    )
    frequency: str = Field(
        description="How often to take it e.g. once daily, twice daily"
    )


class JargonItem(BaseModel):
    """
    One medical term explained in plain English.
    source_sentence is the exact sentence from the document
    where this term appeared — used for hallucination grounding.
    """
    term: str = Field(description="The medical term exactly as it appears in the document")
    explanation: str = Field(
        description="Plain English explanation, grade 6-8 reading level, 1-2 sentences"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident the model is in this explanation, 0 to 1"
    )
    source_sentence: str = Field(
        description="The sentence from the original document where this term appeared"
    )


class MedBridgeOutput(BaseModel):
    """
    The complete output of one MedBridge analysis.
    This is what gets returned from the API and
    displayed in the frontend.
    """
    summary: str = Field(
        description="3 sentences max. Plain English. Grade 6-8 reading level. "
                    "What happened, what was done, what the patient needs to know."
    )
    urgency_flags: List[UrgencyFlag] = Field(
        default=[],
        description="Things the patient needs to act on. Empty list if none."
    )
    medications: List[MedicationItem] = Field(
        default=[],
        description="All medications mentioned. Empty list for non-discharge docs."
    )
    jargon: List[JargonItem] = Field(
        default=[],
        description="Medical terms explained in plain English."
    )
    questions: List[str] = Field(
        description="3 to 5 questions the patient should ask their doctor. "
                    "Generated from the specific content of this document."
    )
    doc_type: Literal["discharge", "lab", "radiology", "eob"] = Field(
        description="Type of document that was analyzed"
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the analysis, 0 to 1"
    )


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # test that the schema works by creating a sample output
    sample = MedBridgeOutput(
        summary="You were admitted for chest pain and found to have a mild heart attack. "
                "A small metal tube called a stent was placed to open a blocked artery. "
                "You need to take blood thinning medications every day.",
        urgency_flags=[
            UrgencyFlag(
                text="Follow up with your cardiologist",
                timeframe="within 48 hours",
                severity="critical"
            ),
            UrgencyFlag(
                text="Do not stop aspirin or clopidogrel",
                timeframe="ongoing",
                severity="critical"
            )
        ],
        medications=[
            MedicationItem(
                name="clopidogrel",
                brand_name="Plavix",
                purpose="Prevents blood clots from forming around your new stent",
                warning="Do not stop taking this without talking to your cardiologist first",
                frequency="once daily"
            ),
            MedicationItem(
                name="aspirin",
                brand_name="Bayer",
                purpose="Keeps blood from clotting, works together with clopidogrel",
                warning="Do not stop without consulting your doctor",
                frequency="once daily, 81mg"
            )
        ],
        jargon=[
            JargonItem(
                term="NSTEMI",
                explanation="A type of heart attack where one artery is partially blocked. "
                            "Less severe than a full blockage but still serious and requires treatment.",
                confidence=0.97,
                source_sentence="Final Diagnosis: Non-ST elevation myocardial infarction (NSTEMI)"
            ),
            JargonItem(
                term="Percutaneous coronary intervention",
                explanation="A non-surgical procedure to open a blocked heart artery using "
                            "a tiny balloon and metal mesh tube (stent), done through a small "
                            "cut in your wrist or groin.",
                confidence=0.95,
                source_sentence="Procedure: Percutaneous coronary intervention (PCI)"
            )
        ],
        questions=[
            "How long do I need to take both aspirin and clopidogrel?",
            "What warning signs should make me call 911 right away?",
            "When can I return to normal physical activity?",
            "What foods or activities should I avoid while my heart heals?",
            "What does my echocardiogram in 4-6 weeks check for?"
        ],
        doc_type="discharge",
        overall_confidence=0.91
    )

    print("Schema validation passed.")
    print(f"\nSummary: {sample.summary}")
    print(f"\nUrgency flags: {len(sample.urgency_flags)}")
    print(f"Medications:   {len(sample.medications)}")
    print(f"Jargon terms:  {len(sample.jargon)}")
    print(f"Questions:     {len(sample.questions)}")
    print(f"Doc type:      {sample.doc_type}")
    print(f"Confidence:    {sample.overall_confidence}")
    print(f"\nJSON output preview:")
    print(sample.model_dump_json(indent=2)[:500])