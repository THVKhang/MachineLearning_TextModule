"""
Build comparison bar chart: TF-IDF vs SBERT Embedding (F1-weighted).

Usage:
    python scripts/build_comparison_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

RESULT_TABLES = PROJECT_ROOT / "results" / "tables"
RESULT_FIGURES = PROJECT_ROOT / "results" / "figures"


def build_comparison_figure(save_path: Path | None = None) -> str:
    """Build and save TF-IDF vs Embedding comparison bar chart.

    Returns the path to the saved figure.
    """
    tfidf_path = RESULT_TABLES / "tfidf_model_comparison.csv"
    bert_path = RESULT_TABLES / "bert_benchmark_results.csv"

    if not tfidf_path.exists() or not bert_path.exists():
        raise FileNotFoundError(
            f"Required CSVs missing. Expected:\n  {tfidf_path}\n  {bert_path}"
        )

    tfidf_df = pd.read_csv(tfidf_path)
    bert_df = pd.read_csv(bert_path)

    # --- TF-IDF rows ---
    tfidf_df["branch"] = "TF-IDF"
    tfidf_df["display_label"] = (
        tfidf_df["model"].str.replace("_", " ").str.title()
        + "\n(TF-IDF · "
        + tfidf_df["tfidf_config"]
        + " · full)"
    )
    tfidf_data = tfidf_df[["display_label", "f1_weighted", "branch"]].rename(
        columns={"f1_weighted": "f1"}
    )

    # --- Embedding rows ---
    bert_df["F1-Score"] = bert_df["F1-Score"].astype(float)
    bert_df["branch"] = "SBERT Embedding"
    bert_df["display_label"] = (
        bert_df["Model"].str.replace("_", " ").str.title()
        + "\n(SBERT · "
        + bert_df["Dataset"]
        + ")"
    )
    bert_data = bert_df[["display_label", "F1-Score", "branch"]].rename(
        columns={"F1-Score": "f1"}
    )

    combined = pd.concat([tfidf_data, bert_data], ignore_index=True).sort_values(
        "f1", ascending=True
    )

    # --- Plot ---
    color_map = {"TF-IDF": "#3B82F6", "SBERT Embedding": "#F59E0B"}
    colors = [color_map[b] for b in combined["branch"]]

    fig, ax = plt.subplots(figsize=(13, max(7, len(combined) * 0.6)))
    bars = ax.barh(
        combined["display_label"],
        combined["f1"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.65,
    )

    for bar, val in zip(bars, combined["f1"]):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("F1-weighted (higher is better)", fontsize=12)
    ax.set_title(
        "TF-IDF vs SBERT Embedding — F1-weighted Comparison\n"
        "Dataset: AG News · 4 classes · Primary metric: F1-weighted",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0.50, 1.00)

    ax.axvline(0.87, color="red", linestyle="--", alpha=0.75, linewidth=1.5)
    ax.axvline(0.90, color="green", linestyle="--", alpha=0.75, linewidth=1.5)

    tfidf_patch = mpatches.Patch(color="#3B82F6", label="TF-IDF branch")
    emb_patch = mpatches.Patch(color="#F59E0B", label="SBERT Embedding branch")
    thresh_line = plt.Line2D([0], [0], color="red", linestyle="--", label="Threshold 0.87")
    target_line = plt.Line2D([0], [0], color="green", linestyle="--", label="Target 0.90")
    ax.legend(
        handles=[tfidf_patch, emb_patch, thresh_line, target_line],
        loc="lower right",
        fontsize=9,
    )

    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    plt.tight_layout()

    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = RESULT_FIGURES / "comparison_tfidf_vs_embedding.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] Comparison chart saved: {save_path}")
    return str(save_path)


def main() -> None:
    build_comparison_figure()


if __name__ == "__main__":
    main()
