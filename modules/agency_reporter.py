from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def run_reporter(
    project_root: Path,
    plan: dict[str, Any],
    tfidf_run: dict[str, Any] | None,
    embedding_run: dict[str, Any] | None,
    critic_report: dict[str, Any],
) -> dict[str, str]:
    reports_dir = project_root / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    verdict = critic_report.get("status", "unknown")
    best_family = critic_report.get("best_family", "unknown")
    best_model = critic_report.get("best_model", "unknown")
    best_metric = _fmt_float(critic_report.get("best_metric"))
    primary_metric = critic_report.get("primary_metric", "f1_weighted")

    recommendation = "Use current best model for final report."
    if verdict == "fail":
        recommendation = "Do not finalize yet. Resolve critic errors and re-run workflow."
    elif verdict == "warn":
        recommendation = "Finalize with caution and document known gaps in report limitations."

    lines = [
        "# Agency Workflow Summary",
        "",
        "## Plan",
        f"- Objective: {plan.get('objective', 'n/a')}",
        f"- Mode: {plan.get('mode', 'n/a')}",
        f"- Feature Family: {plan.get('feature_family', 'n/a')}",
        f"- Recommended Model: {plan.get('recommended_model', 'n/a')}",
        f"- Primary Metric: {plan.get('primary_metric', 'n/a')}",
        f"- Benchmark Scales: {plan.get('benchmark_scales', '')}",
        "",
        "## Runner Outputs",
        f"- TF-IDF log: {(tfidf_run or {}).get('log_path', 'not-run')}",
        f"- Embedding log: {(embedding_run or {}).get('log_path', 'not-run')}",
        "",
        "## Critic Verdict",
        f"- Status: {verdict}",
        f"- Best Family: {best_family}",
        f"- Best Model: {best_model}",
        f"- Best {primary_metric}: {best_metric}",
        f"- Threshold: {_fmt_float(critic_report.get('threshold'))}",
        "",
        "## Recommendation",
        f"- {recommendation}",
    ]

    findings = critic_report.get("findings", [])
    if findings:
        lines.extend(["", "## Findings"])
        for finding in findings:
            lines.append(f"- [{finding.get('severity', 'info')}] {finding.get('code', 'finding')}: {finding.get('message', '')}")

    md_path = reports_dir / "agency_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "plan": plan,
        "tfidf_run": tfidf_run,
        "embedding_run": embedding_run,
        "critic": critic_report,
        "recommendation": recommendation,
        "summary_markdown": str(md_path),
    }
    json_path = reports_dir / "agency_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "summary_markdown": str(md_path),
        "summary_json": str(json_path),
        "recommendation": recommendation,
    }
