"""
run_eval.py
-----------
Runs the full evaluation harness across MIMIC-IV documents.

What it does:
1. Loads N documents from mimic-iv-bhc.csv
2. Runs each through the full MedBridge pipeline
3. Evaluates each output with readability, grounding, judge
4. Logs all metrics to MLflow
5. Prints a final summary table

Usage:
    python run_eval.py --n 10    # run on 10 documents (quick test)
    python run_eval.py --n 50    # run on 50 documents (full eval)

Takes about 30-60 seconds per document due to Gemini API calls.
50 documents = ~40 minutes total.
"""

import os
import sys
import argparse
import time
import pandas as pd
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from medbridge.pipeline import run_pipeline_from_bytes
from medbridge.eval.readability import evaluate_summary
from medbridge.eval.grounding import evaluate_grounding, evaluate_batch_grounding
from medbridge.eval.judge import judge_output, evaluate_batch_judge

MIMIC_PATH = os.path.join("data", "samples", "mimic-iv-bhc.csv")


def load_mimic_samples(n: int = 50) -> list:
    """
    Loads N discharge notes from MIMIC-IV-BHC dataset.
    Filters for notes with reasonable length.
    """
    print(f"Loading {n} samples from MIMIC-IV...")
    df = pd.read_csv(MIMIC_PATH, nrows=n * 3)

    df = df[
        (df["input_tokens"] >= 200) &
        (df["input_tokens"] <= 4000) &
        (df["target_tokens"] >= 50)
    ].head(n)

    print(f"Loaded {len(df)} samples after filtering")
    return df.to_dict("records")


def text_to_pdf_bytes(text: str) -> bytes:
    """
    Converts plain text to a PDF for pipeline input.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()
    story = []

    for para in text.split("\n"):
        if para.strip():
            try:
                story.append(Paragraph(para.strip(), styles["Normal"]))
            except Exception:
                pass

    if not story:
        story.append(Paragraph("Medical document", styles["Normal"]))

    doc.build(story)
    return buffer.getvalue()


def run_evaluation(n: int = 10, experiment_name: str = "medbridge-eval"):
    """
    Main evaluation function.
    """
    samples = load_mimic_samples(n)

    if not samples:
        print("ERROR: No samples loaded. Check MIMIC CSV path.")
        return

    mlflow.set_experiment(experiment_name)

    readability_results = []
    grounding_results = []
    judge_results = []
    pipeline_errors = 0

    with mlflow.start_run(run_name=f"eval_{n}_docs"):
        mlflow.log_param("n_documents", n)
        mlflow.log_param("dataset", "MIMIC-IV-Ext-BHC")
        mlflow.log_param("model", "gemini-2.5-flash")

        print(f"\nRunning evaluation on {len(samples)} documents...\n")

        for i, sample in enumerate(samples):
            note_id = sample.get("note_id", f"doc_{i}")
            input_text = sample.get("input", "")

            print(f"[{i+1}/{len(samples)}] Processing note {note_id}...")

            try:
                # convert text to PDF bytes
                pdf_bytes = text_to_pdf_bytes(input_text)

                # run full pipeline
                start = time.time()
                result = run_pipeline_from_bytes(pdf_bytes)
                elapsed = time.time() - start

                print(f"  Pipeline: {elapsed:.1f}s | Type: {result.doc_type} | Conf: {result.overall_confidence:.0%}")

                # 1. readability
                read_result = evaluate_summary(result.summary)
                readability_results.append(read_result)
                print(f"  FK Grade: {read_result['fk_grade']} ({'PASS' if read_result['passed'] else 'FAIL'})")

                # 2. grounding
                jargon_dicts = [j.model_dump() for j in result.jargon]
                ground_result = evaluate_grounding(jargon_dicts, input_text)
                grounding_results.append(ground_result)
                rate_pct = ground_result.get("grounding_rate_pct", "N/A")
                passed = ground_result.get("passed", False)
                print(f"  Grounding: {rate_pct} ({'PASS' if passed else 'FAIL'})")

                # 3. judge
                med_dicts = [m.model_dump() for m in result.medications]
                q_list = list(result.questions)
                judge_result = judge_output(
                    input_text,
                    result.summary,
                    med_dicts,
                    jargon_dicts,
                    q_list
                )
                judge_results.append(judge_result)
                print(f"  Judge: {judge_result['overall']}/5 ({'PASS' if judge_result['passed'] else 'FAIL'})")

                # log per-document metrics
                mlflow.log_metrics({
                    f"fk_grade_{i}": read_result["fk_grade"],
                    f"grounding_{i}": ground_result.get("grounding_rate", 0),
                    f"judge_{i}": judge_result["overall"],
                }, step=i)

            except Exception as e:
                print(f"  ERROR: {e}")
                pipeline_errors += 1
                continue

            print()

        # ── AGGREGATE RESULTS ────────────────────────────────────
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        # readability
        if readability_results:
            grades = [r["fk_grade"] for r in readability_results]
            avg_grade = round(sum(grades) / len(grades), 2)
            pass_count = sum(1 for r in readability_results if r["passed"])
            pass_rate = round(pass_count / len(readability_results) * 100, 1)

            print(f"\nREADABILITY")
            print(f"  Avg FK Grade:  {avg_grade}")
            print(f"  Pass Rate:     {pass_rate}%")
            print(f"  Min Grade:     {min(grades)}")
            print(f"  Max Grade:     {max(grades)}")

            mlflow.log_metrics({
                "avg_fk_grade": avg_grade,
                "readability_pass_rate": pass_rate,
            })

        # grounding
        if grounding_results:
            ground_agg = evaluate_batch_grounding(grounding_results)
            print(f"\nGROUNDING")
            print(f"  Avg Rate:      {ground_agg.get('avg_grounding_rate_pct', 'N/A')}")
            print(f"  Pass Rate:     {ground_agg.get('pass_rate', 'N/A')}%")

            mlflow.log_metrics({
                "avg_grounding_rate": ground_agg.get("avg_grounding_rate", 0),
                "grounding_pass_rate": ground_agg.get("pass_rate", 0),
            })

        # judge
        if judge_results:
            judge_agg = evaluate_batch_judge(judge_results)
            print(f"\nJUDGE SCORES")
            print(f"  Avg Clarity:      {judge_agg.get('avg_clarity', 0)}/5")
            print(f"  Avg Accuracy:     {judge_agg.get('avg_accuracy', 0)}/5")
            print(f"  Avg Completeness: {judge_agg.get('avg_completeness', 0)}/5")
            print(f"  Avg Overall:      {judge_agg.get('avg_overall', 0)}/5")
            print(f"  Pass Rate:        {judge_agg.get('pass_rate', 0)}%")

            mlflow.log_metrics({
                "avg_judge_clarity": judge_agg.get("avg_clarity", 0),
                "avg_judge_accuracy": judge_agg.get("avg_accuracy", 0),
                "avg_judge_completeness": judge_agg.get("avg_completeness", 0),
                "avg_judge_overall": judge_agg.get("avg_overall", 0),
                "judge_pass_rate": judge_agg.get("pass_rate", 0),
            })

        print(f"\nPIPELINE ERRORS: {pipeline_errors}/{len(samples)}")
        mlflow.log_param("pipeline_errors", pipeline_errors)

        print(f"\nMLflow run complete.")
        print(f"View results: mlflow ui")
        print(f"Then open:    http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedBridge evaluation")
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of documents to evaluate (default: 5)"
    )
    parser.add_argument(
        "--experiment", type=str, default="medbridge-eval",
        help="MLflow experiment name"
    )
    args = parser.parse_args()
    run_evaluation(n=args.n, experiment_name=args.experiment)