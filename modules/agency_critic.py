from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CriticFinding:
    severity: str
    code: str
    message: str


@dataclass
class CriticReport:
    status: str
    primary_metric: str
    threshold: float
    best_family: str | None
    best_model: str | None
    best_metric: float | None
    findings: list[CriticFinding]
    checked_files: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "primary_metric": self.primary_metric,
            "threshold": self.threshold,
            "best_family": self.best_family,
            "best_model": self.best_model,
            "best_metric": self.best_metric,
            "findings": [asdict(x) for x in self.findings],
            "checked_files": self.checked_files,
        }


def run_critic(
    project_root: Path,
    primary_metric: str = "f1_weighted",
    threshold: float = 0.85,
    required_scales: tuple[str, ...] = ("5k_2k", "20k_2k"),
) -> dict[str, object]:
    tables_dir = project_root / "results" / "tables"
    results_dir = project_root / "results"

    tfidf_path = tables_dir / "tfidf_model_comparison.csv"
    embedding_path = tables_dir / "bert_benchmark_results.csv"
    compare_path = tables_dir / "feature_family_comparison.csv"

    findings: list[CriticFinding] = []
    checked_files: list[str] = [str(tfidf_path), str(embedding_path), str(compare_path)]

    required_artifacts = [
        results_dir / "cm_logistic_regression.png",
        results_dir / "cm_svm.png",
        results_dir / "cm_naive_bayes.png",
    ]

    for artifact in required_artifacts:
        checked_files.append(str(artifact))
        if not artifact.exists():
            findings.append(
                CriticFinding(
                    severity="error",
                    code="missing_artifact",
                    message=f"Missing required artifact: {artifact}",
                )
            )

    tfidf_best = None
    if tfidf_path.exists():
        tfidf_df = pd.read_csv(tfidf_path)
        if tfidf_df.empty:
            findings.append(CriticFinding("error", "empty_tfidf", "TF-IDF table is empty."))
        else:
            if primary_metric not in tfidf_df.columns:
                findings.append(
                    CriticFinding(
                        "error",
                        "missing_primary_metric",
                        f"Primary metric column '{primary_metric}' not found in TF-IDF table.",
                    )
                )
            else:
                tfidf_best = tfidf_df.sort_values(primary_metric, ascending=False).iloc[0]
    else:
        findings.append(CriticFinding("error", "missing_tfidf_table", f"Missing table: {tfidf_path}"))

    emb_best = None
    if embedding_path.exists():
        emb_df = pd.read_csv(embedding_path)
        if emb_df.empty:
            findings.append(CriticFinding("error", "empty_embedding", "Embedding benchmark table is empty."))
        else:
            if "Dataset" not in emb_df.columns:
                findings.append(CriticFinding("error", "missing_scale_column", "Embedding table lacks Dataset column."))
            else:
                seen_scales = set(str(x) for x in emb_df["Dataset"].dropna().unique())
                missing_scales = [s for s in required_scales if s not in seen_scales]
                if missing_scales:
                    findings.append(
                        CriticFinding(
                            "error",
                            "missing_scales",
                            f"Missing required embedding scales: {', '.join(missing_scales)}",
                        )
                    )

            if "F1-Score" not in emb_df.columns:
                findings.append(CriticFinding("error", "missing_embedding_f1", "Embedding table lacks F1-Score."))
            else:
                emb_df["F1-Score"] = emb_df["F1-Score"].astype(float)
                emb_best = emb_df.sort_values("F1-Score", ascending=False).iloc[0]
    else:
        findings.append(CriticFinding("error", "missing_embedding_table", f"Missing table: {embedding_path}"))

    best_family: str | None = None
    best_model: str | None = None
    best_metric: float | None = None

    if tfidf_best is not None:
        tfidf_metric = float(tfidf_best[primary_metric])
        best_family = "tfidf"
        best_model = str(tfidf_best["model"])
        best_metric = tfidf_metric

    if emb_best is not None:
        emb_metric = float(emb_best["F1-Score"])
        if best_metric is None or emb_metric > best_metric:
            best_family = "embedding"
            best_model = str(emb_best["Model"])
            best_metric = emb_metric

    if best_metric is None:
        findings.append(CriticFinding("error", "no_valid_results", "No valid model results found to evaluate."))
    elif best_metric < threshold:
        findings.append(
            CriticFinding(
                "warning",
                "below_threshold",
                f"Best metric {best_metric:.4f} is below threshold {threshold:.4f}.",
            )
        )

    has_error = any(x.severity == "error" for x in findings)
    has_warning = any(x.severity == "warning" for x in findings)
    status = "fail" if has_error else ("warn" if has_warning else "pass")

    report = CriticReport(
        status=status,
        primary_metric=primary_metric,
        threshold=threshold,
        best_family=best_family,
        best_model=best_model,
        best_metric=best_metric,
        findings=findings,
        checked_files=checked_files,
    )

    logs_dir = project_root / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "critic_report.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    output = report.to_dict()
    output["report_path"] = str(out_path)
    return output
