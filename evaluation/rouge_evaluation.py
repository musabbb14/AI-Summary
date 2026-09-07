"""
evaluation/rouge_evaluation.py

Implements the quantitative half of Objective 4: scoring system-generated
summaries against human-written reference summaries using ROUGE-1,
ROUGE-2, and ROUGE-L (Lin, 2004).

Usage:
    python evaluation/rouge_evaluation.py --data path/to/samples.csv

Expected CSV columns: `document`, `reference_summary`
(e.g. a sample drawn from the CNN/DailyMail test set, per the proposal).

Run from the project root so the `utils` package is importable:
    python -m evaluation.rouge_evaluation --data datasets/sample_documents/samples.csv
"""

import argparse
import csv
import statistics
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rouge_score import rouge_scorer
from utils.summarizer import summarize_long_text


def load_samples(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("document") and row.get("reference_summary")]


def evaluate(csv_path, output_csv):
    samples = load_samples(csv_path)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    for i, sample in enumerate(samples, start=1):
        print(f"Scoring sample {i}/{len(samples)}...")
        result = summarize_long_text(sample["document"])
        generated = result["summary"]
        scores = scorer.score(sample["reference_summary"], generated)

        rows.append({
            "sample_id": i,
            "rouge1_f": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f": round(scores["rougeL"].fmeasure, 4),
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "rouge1_f", "rouge2_f", "rougeL_f"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        print("\n--- Averages ---")
        for key in ["rouge1_f", "rouge2_f", "rougeL_f"]:
            avg = statistics.mean(r[key] for r in rows)
            print(f"{key}: {avg:.4f}")

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summaries with ROUGE metrics.")
    parser.add_argument("--data", required=True, help="CSV with 'document' and 'reference_summary' columns.")
    parser.add_argument("--out", default="evaluation/rouge_results.csv", help="Where to write per-sample scores.")
    args = parser.parse_args()

    evaluate(args.data, args.out)
