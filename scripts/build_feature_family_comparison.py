from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_TABLES = PROJECT_ROOT / "results" / "tables"


def load_best_rows(include_all_embedding_scales: bool = False) -> pd.DataFrame:
    tfidf_path = RESULT_TABLES / "tfidf_model_comparison.csv"
    bert_path = RESULT_TABLES / "bert_benchmark_results.csv"

    tfidf_df = pd.read_csv(tfidf_path)
    bert_df = pd.read_csv(bert_path)

    tfidf_best = tfidf_df.sort_values("f1_weighted", ascending=False).head(1).copy()
    tfidf_best["feature_family"] = "tfidf"
    tfidf_best["benchmark_scale"] = tfidf_best["tfidf_config"]
    tfidf_best["benchmark_type"] = "tfidf_config"
    tfidf_best["feature_dim"] = pd.NA
    tfidf_best["embed_time_sec"] = pd.NA
    tfidf_best["comparison_note"] = "best TF-IDF config"

    if include_all_embedding_scales:
        # Keep the best model per embedding scale to expose scale-level tradeoffs.
        bert_df["F1-Score"] = bert_df["F1-Score"].astype(float)
        bert_rows = (
            bert_df.sort_values(["Dataset", "F1-Score"], ascending=[True, False])
            .groupby("Dataset", as_index=False)
            .head(1)
            .copy()
        )
        bert_rows["feature_family"] = "embedding"
        bert_rows["benchmark_scale"] = bert_rows["Dataset"]
        bert_rows["benchmark_type"] = "embedding_scale"
        bert_rows["feature_dim"] = bert_rows["Embedding Dim"]
        bert_rows["embed_time_sec"] = bert_rows["Embed Time (s)"]
        bert_rows["comparison_note"] = "best embedding model per scale"
    else:
        bert_rows = bert_df.sort_values("F1-Score", ascending=False).head(1).copy()
        bert_rows["feature_family"] = "embedding"
        bert_rows["benchmark_scale"] = bert_rows["Dataset"]
        bert_rows["benchmark_type"] = "embedding_scale"
        bert_rows["feature_dim"] = bert_rows["Embedding Dim"]
        bert_rows["embed_time_sec"] = bert_rows["Embed Time (s)"]
        bert_rows["comparison_note"] = "best embedding config"

    tfidf_best = tfidf_best.rename(
        columns={
            "model": "model",
            "train_size": "train_size",
            "test_size": "test_size",
            "accuracy": "accuracy",
            "precision_weighted": "precision",
            "recall_weighted": "recall",
            "f1_weighted": "f1",
        }
    )
    tfidf_best["accuracy"] = tfidf_best["accuracy"].astype(float)
    tfidf_best["precision"] = tfidf_best["precision"].astype(float)
    tfidf_best["recall"] = tfidf_best["recall"].astype(float)
    tfidf_best["f1"] = tfidf_best["f1"].astype(float)
    tfidf_best["train_time_sec"] = tfidf_best["train_time_sec"].astype(float)

    bert_rows = bert_rows.rename(columns={"Model": "model", "Train Size": "train_size", "Test Size": "test_size", "Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1-Score": "f1"})
    bert_rows["accuracy"] = bert_rows["accuracy"].astype(float)
    bert_rows["precision"] = bert_rows["precision"].astype(float)
    bert_rows["recall"] = bert_rows["recall"].astype(float)
    bert_rows["f1"] = bert_rows["f1"].astype(float)
    bert_rows["train_time_sec"] = bert_rows["Train Time (s)"].astype(float)

    common_columns = [
        "feature_family",
        "benchmark_type",
        "benchmark_scale",
        "model",
        "train_size",
        "test_size",
        "feature_dim",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train_time_sec",
        "embed_time_sec",
        "comparison_note",
    ]

    tfidf_out = tfidf_best[common_columns].copy()
    bert_out = bert_rows[common_columns].copy()
    out = pd.concat([tfidf_out, bert_out], ignore_index=True)
    out = out.sort_values(["f1", "train_time_sec"], ascending=[False, True])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TF-IDF vs embedding comparison tables.")
    parser.add_argument(
        "--include-all-embedding-scales",
        action="store_true",
        help="Keep best embedding row for each scale instead of a single global best.",
    )
    parser.add_argument(
        "--output-name",
        default="feature_family_comparison.csv",
        help="Output CSV name under results/tables/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    out = load_best_rows(include_all_embedding_scales=args.include_all_embedding_scales)
    out_path = RESULT_TABLES / args.output_name
    out.to_csv(out_path, index=False)
    print(f"Saved comparison table to: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
