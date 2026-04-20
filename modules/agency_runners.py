from __future__ import annotations

from typing import Any

from bert_benchmark import run_embedding_benchmark
from run_experiments import run_tfidf_benchmark


def run_tfidf_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    recommended_cfg = str(plan.get("recommended_feature_config", ""))
    selected_configs = None
    tfidf_candidates = [part.strip() for part in recommended_cfg.split("+")]
    tfidf_candidates = [part for part in tfidf_candidates if part.startswith("tfidf_")]
    if tfidf_candidates:
        selected_configs = [tfidf_candidates[0]]

    return run_tfidf_benchmark(
        mode=str(plan.get("mode", "demo")),
        config_names=selected_configs,
        model_list=None,
        primary_metric=str(plan.get("primary_metric", "f1_weighted")),
    )


def run_embedding_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    scales_raw = str(plan.get("benchmark_scales", "")).strip()
    scales = [x.strip() for x in scales_raw.split(",") if x.strip()]
    if not scales:
        scales = ["5k_2k"]

    return run_embedding_benchmark(scales=scales, model_list=None)
