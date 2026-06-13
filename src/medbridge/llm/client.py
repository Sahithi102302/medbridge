"""
client.py
---------
Step 5 of the MedBridge pipeline.

Job: Take the prompt + document text, call Gemini API,
validate the response against our Pydantic schema,
return a clean MedBridgeOutput object.

Why this exists:
- Gemini is an external API — it can fail, timeout, return bad JSON
- We need retry logic so one failure doesn't crash everything
- We need Pydantic validation so bad output is caught immediately
- We abstract the API call so switching to Claude/OpenAI is one line

What happens here:
1. Load API key from .env
2. Build the prompt using prompt_templates.py
3. Call Gemini API
4. Parse the JSON response
5. Validate against MedBridgeOutput schema
6. If validation fails — retry with error correction prompt
7. Return clean MedBridgeOutput object
"""

import os
import json
import re
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from medbridge.llm.schemas import MedBridgeOutput
from medbridge.llm.prompt_templates import get_system_prompt, build_user_message

load_dotenv()

# configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# use gemini-1.5-flash — fast and free tier friendly
# easy to switch to gemini-1.5-pro for better quality
MODEL_NAME = "gemini-2.5-flash"

# generation config — temperature 0 for consistent output
GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.0,        # deterministic output
    max_output_tokens=3000, # enough for full structured response
)


def extract_json_from_response(text: str) -> str:
    """
    Extracts JSON from Gemini response.
    Sometimes Gemini wraps JSON in markdown code blocks
    like ```json ... ``` — this strips those out.

    Args:
        text: raw response text from Gemini

    Returns:
        clean JSON string
    """
    # try to find JSON in code blocks first
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block:
        return code_block.group(1).strip()

    # try to find raw JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group(0).strip()

    # return as-is and let json.loads handle the error
    return text.strip()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_gemini(system_prompt: str, user_message: str) -> str:
    """
    Calls the Gemini API with retry logic.
    Retries up to 3 times with exponential backoff
    if the API fails or times out.

    Args:
        system_prompt: the system instructions
        user_message:  the document text and context

    Returns:
        raw text response from Gemini
    """
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_prompt,
        generation_config=GENERATION_CONFIG,
    )

    response = model.generate_content(user_message)
    return response.text


def analyze_document(
    text: str,
    entities: list,
    sections: list,
    doc_type: str,
) -> MedBridgeOutput:
    """
    Main function. Takes NLP pipeline output, calls Gemini,
    returns validated MedBridgeOutput.

    Args:
        text:     clean text from parser.py
        entities: medical entities from ner.py
        sections: document sections from chunker.py
        doc_type: document type from classifier.py

    Returns:
        MedBridgeOutput — validated Pydantic object

    Raises:
        ValueError: if Gemini response cannot be parsed after retries
    """
    # get the right prompt template for this doc type
    system_prompt = get_system_prompt(doc_type)

    # build the user message with document context
    user_message = build_user_message(text, entities, sections, doc_type)

    print(f"  Calling Gemini ({MODEL_NAME})...")

    # call Gemini
    raw_response = call_gemini(system_prompt, user_message)

    # extract JSON from response
    json_str = extract_json_from_response(raw_response)

    # parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # try error correction — ask Gemini to fix the JSON
        print(f"  JSON parse failed: {e}")
        print("  Retrying with error correction prompt...")
        correction_prompt = f"""Your previous response was not valid JSON.
Error: {e}

Please return ONLY valid JSON matching this structure:
{{
    "summary": "string",
    "urgency_flags": [],
    "medications": [],
    "jargon": [],
    "questions": ["q1", "q2", "q3"],
    "doc_type": "{doc_type}",
    "overall_confidence": 0.8
}}

Previous response:
{raw_response[:500]}"""

        raw_response = call_gemini(system_prompt, correction_prompt)
        json_str = extract_json_from_response(raw_response)
        data = json.loads(json_str)

    # ensure doc_type is set correctly
    data["doc_type"] = doc_type

    # validate against Pydantic schema
    try:
        output = MedBridgeOutput(**data)
    except Exception as e:
        print(f"  Schema validation failed: {e}")
        # create a minimal valid output rather than crashing
        output = MedBridgeOutput(
            summary=data.get("summary", "Analysis could not be completed."),
            urgency_flags=data.get("urgency_flags", []),
            medications=data.get("medications", []),
            jargon=data.get("jargon", []),
            questions=data.get("questions", [
                "Please ask your doctor to explain this document.",
            ]),
            doc_type=doc_type,
            overall_confidence=0.5,
        )

    return output


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from medbridge.ingest.parser import extract_text_from_pdf
    from medbridge.nlp.ner import extract_medical_entities
    from medbridge.nlp.chunker import split_into_sections
    from medbridge.nlp.classifier import classify_document

    sample_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "samples"
    )

    # test on discharge summary only first
    pdf_path = os.path.join(sample_dir, "discharge_summary.pdf")

    print("=== Testing client.py — full pipeline ===\n")
    print("Step 1: Parsing PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(text)} characters")

    print("Step 2: Classifying document...")
    doc_type, confidence = classify_document(text)
    print(f"  Type: {doc_type} ({confidence:.2%})")

    print("Step 3: Extracting medical entities...")
    entities = extract_medical_entities(text)
    print(f"  Found {len(entities)} entities")

    print("Step 4: Splitting into sections...")
    sections = split_into_sections(text)
    print(f"  Found {len(sections)} sections")

    print("Step 5: Calling Gemini API...")
    result = analyze_document(text, entities, sections, doc_type)

    print("\n=== RESULTS ===\n")
    print(f"Summary:\n{result.summary}\n")
    print(f"Urgency flags: {len(result.urgency_flags)}")
    for flag in result.urgency_flags:
        print(f"  [{flag.severity.upper()}] {flag.text} ({flag.timeframe})")

    print(f"\nMedications: {len(result.medications)}")
    for med in result.medications:
        print(f"  {med.name} ({med.brand_name}) — {med.purpose}")

    print(f"\nJargon terms: {len(result.jargon)}")
    for j in result.jargon:
        print(f"  {j.term} ({j.confidence:.0%}): {j.explanation[:60]}...")

    print(f"\nDoctor questions: {len(result.questions)}")
    for i, q in enumerate(result.questions, 1):
        print(f"  {i}. {q}")

    print(f"\nOverall confidence: {result.overall_confidence}")
    print("\nSTATUS: OK")