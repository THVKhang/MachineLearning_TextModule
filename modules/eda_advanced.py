"""
Advanced EDA module: noisy/outlier detection and classification error analysis.

Produces report-ready JSON + CSV outputs under results/eda/.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_stats(texts: list[str]) -> pd.DataFrame:
    records = []
    for i, text in enumerate(texts):
        words = text.split()
        url_count = len(re.findall(r"https?://\S+", text))
        records.append(
            {
                "idx": i,
                "char_len": len(text),
                "word_count": len(words),
                "url_count": url_count,
                "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
                "upper_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
                "is_empty": len(text.strip()) == 0,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Noisy / Outlier Detection
# ---------------------------------------------------------------------------

def detect_noisy_examples(
    texts: list[str],
    labels: list[int] | np.ndarray,
    class_names: list[str] | None = None,
    short_threshold: int = 20,
    long_threshold: int = 1000,
    top_n: int = 10,
) -> dict[str, Any]:
    """Detect noisy or outlier examples from text statistics.

    Checks for: short/long texts, URL-heavy texts, duplicates, high digit ratio,
    and empty texts.
    """
    stats_df = _text_stats(texts)
    labels_arr = np.array(labels)
    stats_df["label"] = labels_arr

    def _row_to_example(row: pd.Series, preview_len: int = 200) -> dict[str, Any]:
        idx = int(row["idx"])
        lbl = int(row["label"])
        return {
            "idx": idx,
            "text_preview": texts[idx][:preview_len],
            "char_len": int(row["char_len"]),
            "label": lbl,
            "class": class_names[lbl] if class_names else str(lbl),
        }

    # Short texts
    short_mask = stats_df["char_len"] < short_threshold
    short_findings = {
        "count": int(short_mask.sum()),
        "threshold_chars": short_threshold,
        "examples": [_row_to_example(r) for _, r in stats_df[short_mask].head(top_n).iterrows()],
    }

    # Long texts
    long_mask = stats_df["char_len"] > long_threshold
    long_findings = {
        "count": int(long_mask.sum()),
        "threshold_chars": long_threshold,
        "examples": [_row_to_example(r) for _, r in stats_df[long_mask].head(top_n).iterrows()],
    }

    # URL-heavy texts
    url_mask = stats_df["url_count"] > 2
    url_findings = {
        "count": int(url_mask.sum()),
        "examples": [
            {**_row_to_example(r), "url_count": int(r["url_count"])}
            for _, r in stats_df[url_mask].head(top_n).iterrows()
        ],
    }

    # Duplicates
    text_counts: Counter = Counter(texts)
    dup_map = {t: c for t, c in text_counts.items() if c > 1}
    dup_findings = {
        "unique_duplicate_texts": len(dup_map),
        "total_duplicate_instances": sum(c - 1 for c in dup_map.values()),
        "top_examples": [
            {"text_preview": t[:150], "count": c}
            for t, c in sorted(dup_map.items(), key=lambda x: -x[1])[:top_n]
        ],
    }

    # High digit ratio
    digit_mask = stats_df["digit_ratio"] > 0.30
    digit_findings = {"count": int(digit_mask.sum()), "threshold": 0.30}

    # Empty texts
    empty_findings = {"count": int(stats_df["is_empty"].sum())}

    noise_total = (
        short_findings["count"]
        + dup_findings["total_duplicate_instances"]
        + empty_findings["count"]
    )
    summary = {
        "total_texts": len(texts),
        "short_texts": short_findings["count"],
        "long_texts": long_findings["count"],
        "duplicates_total": dup_findings["total_duplicate_instances"],
        "url_heavy": url_findings["count"],
        "high_digit_ratio": digit_findings["count"],
        "empty": empty_findings["count"],
        "noise_pct": round(100.0 * noise_total / max(len(texts), 1), 2),
    }

    return {
        "summary": summary,
        "short_texts": short_findings,
        "long_texts": long_findings,
        "url_heavy_texts": url_findings,
        "duplicates": dup_findings,
        "high_digit_ratio": digit_findings,
        "empty_texts": empty_findings,
    }


# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------

def analyze_errors(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    model_name: str = "model",
    top_n: int = 10,
) -> dict[str, Any]:
    """Analyze misclassified examples for one trained model.

    Returns a structured dict with confusion pairs and sample errors.
    """
    mask = y_true != y_pred
    error_indices = np.where(mask)[0]

    sample_errors: list[dict[str, Any]] = []
    for idx in error_indices[:top_n]:
        tl = int(y_true[idx])
        pl = int(y_pred[idx])
        sample_errors.append(
            {
                "idx": int(idx),
                "true_label": tl,
                "pred_label": pl,
                "true_class": class_names[tl] if class_names else str(tl),
                "pred_class": class_names[pl] if class_names else str(pl),
                "text_preview": texts[idx][:250],
                "text_len": len(texts[idx]),
            }
        )

    pair_counts: Counter = Counter(
        (int(y_true[i]), int(y_pred[i])) for i in error_indices
    )
    top_confusion_pairs = [
        {
            "true": class_names[p[0]] if class_names else str(p[0]),
            "pred": class_names[p[1]] if class_names else str(p[1]),
            "count": cnt,
        }
        for p, cnt in pair_counts.most_common(10)
    ]

    # Per-class error rate
    per_class: dict[str, dict[str, Any]] = {}
    for cls_idx, cls_name in enumerate(class_names or [str(i) for i in range(int(y_true.max()) + 1)]):
        cls_mask = y_true == cls_idx
        total = int(cls_mask.sum())
        wrong = int((mask & cls_mask).sum())
        per_class[cls_name] = {
            "total": total,
            "errors": wrong,
            "error_rate": round(wrong / max(total, 1), 4),
        }

    return {
        "model": model_name,
        "total_test": int(len(y_true)),
        "total_errors": int(mask.sum()),
        "error_rate": round(float(mask.sum()) / max(len(y_true), 1), 4),
        "top_confusion_pairs": top_confusion_pairs,
        "per_class_errors": per_class,
        "sample_errors": sample_errors,
    }


# ---------------------------------------------------------------------------
# Run full EDA advanced pipeline
# ---------------------------------------------------------------------------

def run_eda_advanced(
    project_root: Path,
    texts: list[str],
    labels: list[int] | np.ndarray,
    class_names: list[str] | None = None,
    error_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run full advanced EDA: noise detection + error analysis.

    Saves outputs to results/eda/. Returns paths dict for Reporter.
    """
    eda_dir = project_root / "results" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Noise/outlier detection
    noise = detect_noisy_examples(
        texts=texts, labels=labels, class_names=class_names
    )
    noise_path = eda_dir / "noisy_outlier_report.json"
    noise_path.write_text(json.dumps(noise, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary CSV
    summary = noise["summary"]
    summary_csv_path = eda_dir / "noise_summary.csv"
    pd.DataFrame(
        [
            {"metric": "Total texts", "value": summary["total_texts"]},
            {"metric": "Short texts (< 20 chars)", "value": summary["short_texts"]},
            {"metric": "Long texts (> 1000 chars)", "value": summary["long_texts"]},
            {"metric": "Duplicate instances", "value": summary["duplicates_total"]},
            {"metric": "URL-heavy texts (>2 URLs)", "value": summary["url_heavy"]},
            {"metric": "High digit-ratio texts", "value": summary["high_digit_ratio"]},
            {"metric": "Empty texts", "value": summary["empty"]},
            {"metric": "Estimated noise %", "value": f"{summary['noise_pct']}%"},
        ]
    ).to_csv(summary_csv_path, index=False)
    print(f"[EDA] Noise summary saved: {summary_csv_path}")

    output: dict[str, Any] = {
        "noise_report_path": str(noise_path),
        "noise_summary_csv": str(summary_csv_path),
        "noise_summary": summary,
        "error_report_paths": [],
    }

    # Error analysis (list of per-model reports)
    if error_reports:
        error_combined_path = eda_dir / "error_analysis_report.json"
        error_combined_path.write_text(
            json.dumps(error_reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output["error_report_paths"].append(str(error_combined_path))
        print(f"[EDA] Error analysis saved: {error_combined_path}")

    return output
