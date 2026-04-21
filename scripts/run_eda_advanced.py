"""
Standalone script: run advanced EDA (noise detection) on AG News train split.

Usage:
    python scripts/run_eda_advanced.py
    python scripts/run_eda_advanced.py --split test --n 7600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_loader import load_data
from modules.eda_advanced import run_eda_advanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced EDA on AG News dataset.")
    parser.add_argument(
        "--split", choices=["train", "test"], default="train",
        help="Dataset split to analyze (default: train)"
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of samples to analyze (default: all)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[EDA Advanced] Loading ag_news ({args.split})...")
    train_texts, train_labels, test_texts, test_labels, info = load_data("ag_news")

    # AG News class names
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    if args.split == "train":
        texts = list(train_texts)
        labels = list(train_labels)
    else:
        texts = list(test_texts)
        labels = list(test_labels)

    if args.n is not None:
        texts = texts[: args.n]
        labels = labels[: args.n]

    print(f"[EDA Advanced] Analyzing {len(texts):,} {args.split} samples...")
    result = run_eda_advanced(
        project_root=PROJECT_ROOT,
        texts=texts,
        labels=labels,
        class_names=class_names,
    )

    print("\n[EDA Advanced] Noise Summary:")
    summary = result["noise_summary"]
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n[EDA Advanced] Outputs saved:")
    print(f"  Noise report : {result['noise_report_path']}")
    print(f"  Noise CSV    : {result['noise_summary_csv']}")


if __name__ == "__main__":
    main()
