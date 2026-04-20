from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.agency_critic import run_critic
from modules.agency_planner import AgencyPlanner
from modules.agency_reporter import run_reporter
from modules.agency_runners import run_embedding_from_plan, run_tfidf_from_plan
from scripts.build_feature_family_comparison import load_best_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Planner -> Runners -> Critic -> Reporter workflow.")
    parser.add_argument("objective", choices=["fast", "balanced", "best"], help="Workflow objective")
    parser.add_argument(
        "--archive-legacy-bert",
        action="store_true",
        help="Archive legacy features/bert files and old scale folders into features/bert/legacy/.",
    )
    return parser.parse_args()


def archive_legacy_bert_artifacts(project_root: Path) -> list[str]:
    bert_dir = project_root / "features" / "bert"
    moved: list[str] = []
    if not bert_dir.exists():
        return moved

    legacy_targets = [
        bert_dir / "5k",
        bert_dir / "20k",
        bert_dir / "bert_train.npy",
        bert_dir / "bert_test.npy",
        bert_dir / "labels_train.npy",
        bert_dir / "labels_test.npy",
    ]

    to_move = [p for p in legacy_targets if p.exists()]
    if not to_move:
        return moved

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = bert_dir / "legacy" / stamp
    archive_root.mkdir(parents=True, exist_ok=True)

    for src in to_move:
        dst = archive_root / src.name
        shutil.move(str(src), str(dst))
        moved.append(str(dst))

    return moved


def threshold_for_objective(objective: str) -> float:
    if objective == "fast":
        return 0.80
    if objective == "balanced":
        return 0.84
    return 0.87


def main() -> None:
    args = parse_args()
    planner = AgencyPlanner()
    plan = planner.plan(args.objective)
    plan_dict = plan.to_dict()

    print("=== AGENCY WORKFLOW ===")
    print(json.dumps(plan_dict, indent=2))

    archived_paths: list[str] = []
    if args.archive_legacy_bert:
        archived_paths = archive_legacy_bert_artifacts(PROJECT_ROOT)
        if archived_paths:
            print(f"Archived {len(archived_paths)} legacy BERT artifacts.")

    tfidf_run = None
    embedding_run = None

    feature_family = str(plan.feature_family)
    if "tfidf" in feature_family:
        print("\n[Runner] TF-IDF")
        tfidf_run = run_tfidf_from_plan(plan_dict)

    if "embedding" in feature_family:
        print("\n[Runner] Embedding")
        embedding_run = run_embedding_from_plan(plan_dict)

    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Best-vs-best table for final recommendation.
    best_out = load_best_rows(include_all_embedding_scales=False)
    best_path = tables_dir / "feature_family_comparison.csv"
    best_out.to_csv(best_path, index=False)

    # Full scale-aware table to support report analysis.
    full_out = load_best_rows(include_all_embedding_scales=True)
    full_path = tables_dir / "feature_family_comparison_full.csv"
    full_out.to_csv(full_path, index=False)

    critic = run_critic(
        project_root=PROJECT_ROOT,
        primary_metric=str(plan.primary_metric),
        threshold=threshold_for_objective(args.objective),
        required_scales=tuple(plan.benchmark_scales),
    )

    reporter = run_reporter(
        project_root=PROJECT_ROOT,
        plan=plan_dict,
        tfidf_run=tfidf_run,
        embedding_run=embedding_run,
        critic_report=critic,
    )

    workflow_log = {
        "objective": args.objective,
        "plan": plan_dict,
        "tfidf_run": tfidf_run,
        "embedding_run": embedding_run,
        "critic": critic,
        "reporter": reporter,
        "comparison_table": str(best_path),
        "comparison_table_full": str(full_path),
        "archived_legacy_paths": archived_paths,
    }

    logs_dir = PROJECT_ROOT / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    workflow_log_path = logs_dir / "agency_workflow_last.json"
    workflow_log_path.write_text(json.dumps(workflow_log, indent=2), encoding="utf-8")

    print("\n=== WORKFLOW COMPLETE ===")
    print(f"Critic status: {critic['status']}")
    print(f"Recommendation: {reporter['recommendation']}")
    print(f"Workflow log: {workflow_log_path}")


if __name__ == "__main__":
    main()
