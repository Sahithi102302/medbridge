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
import json
import argparse
import time
import pandas as pd
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from medbridge.pipeline import run_pipeline_from_bytes
from medbridge.eval.readability import evaluate_summary
from medbridge.eval.grounding import evaluate_grounding
from medbridge.eval.judge import judge_output
from medbridge.eval.readability import evaluate_batch
from medbridge.eval.grounding import evaluate_batch_grounding
from medbridge.eval.judge import evaluate_batch_judge

# path to MIMIC dataset
MIMIC_PATH = os.path.join("data", "samples", "mimic-iv-bhc.csv")


def load_mimic_samples(n: int = 50) -> list:
    """
    Loads N discharge notes from MIMIC-IV-BHC dataset.
    Filters for notes with reasonable length (500-4000 tokens).
    """
    print(f"Loading {n} samples from MIMIC-IV...")
    df = pd.read_csv(MIMIC_PATH, nrows=n * 3)  # load extra to filter

    # filter for reasonable length
    df = df[
        (df["input_tokens"] >= 200) &
        (df["input_tokens"] <= 4000) &
        (df["target_tokens"] >= 50)
    ].head(n)

    print(f"Loaded {len(df)} samples after filtering")
    return df.to_dict("records")


def text_to_pdf_bytes(text: str) -> bytes:
    """
    Converts plain text to a simple PDF for pipeline input.
    Uses reportlab to create a proper PDF.
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

    # split into paragraphs
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
    Runs pipeline on N MIMIC documents and logs metrics to MLflow.
    """
    # load samples
    samples = load_mimic_samples(n)

    if not samples:
        print("ERROR: No samples loaded. Check MIMIC CSV path.")
        return

    # set up MLflow
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

                # 1. readability evaluation
                read_result = evaluate_summary(result.summary)
                readability_results.append(read_result)
                print(f"  FK Grade: {read_result['fk_grade']} ({'PASS' if read_result['passed'] else 'FAIL'})")

                # 2. grounding evaluation
                jargon_dicts = [j.model_dump() for j in result.jargon]
                ground_result = evaluate_grounding(jargon_dicts, input_text)
                grounding_results.append(ground_result)
                print(f"  Grounding: {ground_result['grounding_rate_pct']} ({'PASS' if ground_result['passed'] else 'FAIL'})")

                # 3. judge evaluation
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
                    f"grounding_{i}": ground_result["grounding_rate"],
                    f"judge_{i}": judge_result["overall"],
                }, step=i)

            except Exception as e:
                print(f"  ERROR: {e}")
                pipeline_errors += 1
                continue

            print()

        # compute aggregate metrics
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        if readability_results:
            read_agg = evaluate_batch([r for r in readability_results])
            print(f"\nREADABILITY")
            print(f"  Avg FK Grade:  {read_agg['avg_fk_grade']}")
            print(f"  Pass Rate:     {read_agg['pass_rate']}%")
            print(f"  Min Grade:     {read_agg['min_fk_grade']}")
            print(f"  Max Grade:     {read_agg['max_fk_grade']}")

            mlflow.log_metrics({
                "avg_fk_grade": read_agg["avg_fk_grade"],
                "readability_pass_rate": read_agg["pass_rate"],
            })

        if grounding_results:
            ground_agg = evaluate_batch_grounding(grounding_results)
            print(f"\nGROUNDING")
            print(f"  Avg Rate:      {ground_agg['avg_grounding_rate_pct']}")
            print(f"  Pass Rate:     {ground_agg['pass_rate']}%")

            mlflow.log_metrics({
                "avg_grounding_rate": ground_agg["avg_grounding_rate"],
                "grounding_pass_rate": ground_agg["pass_rate"],
            })

        if judge_results:
            judge_agg = evaluate_batch_judge(judge_results)
            print(f"\nJUDGE SCORES")
            print(f"  Avg Clarity:      {judge_agg['avg_clarity']}/5")
            print(f"  Avg Accuracy:     {judge_agg['avg_accuracy']}/5")
            print(f"  Avg Completeness: {judge_agg['avg_completeness']}/5")
            print(f"  Avg Overall:      {judge_agg['avg_overall']}/5")
            print(f"  Pass Rate:        {judge_agg['pass_rate']}%")

            mlflow.log_metrics({
                "avg_judge_clarity": judge_agg["avg_clarity"],
                "avg_judge_accuracy": judge_agg["avg_accuracy"],
                "avg_judge_completeness": judge_agg["avg_completeness"],
                "avg_judge_overall": judge_agg["avg_overall"],
                "judge_pass_rate": judge_agg["pass_rate"],
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