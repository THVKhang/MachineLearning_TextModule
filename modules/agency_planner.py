from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

# Planner version
PLANNER_VERSION = "v2"

Objective = Literal["fast", "balanced", "best"]


@dataclass(frozen=True)
class AgencyPlan:
    objective: Objective
    mode: str
    feature_family: str
    model_family: str
    recommended_feature_config: str
    recommended_model: str
    benchmark_scales: tuple[str, ...]
    primary_metric: str
    notes: str
    planner_version: str = PLANNER_VERSION

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["benchmark_scales"] = ", ".join(self.benchmark_scales)
        return data


class AgencyPlanner:
    """Planner v2: selects experiment strategy by objective.

    Changes vs v1:
    - balanced: benchmark_scales upgraded from 5k_2k → 20k_2k
      (20k_2k embedding F1=0.8954 vs 5k_2k F1=0.8736, +2.2%).
    - best notes: includes current benchmark results for reference.
    """

    def plan(self, objective: Objective) -> AgencyPlan:
        if objective == "fast":
            return AgencyPlan(
                objective="fast",
                mode="demo",
                feature_family="tfidf",
                model_family="classical",
                recommended_feature_config="tfidf_uni_1k",
                recommended_model="naive_bayes",
                benchmark_scales=("5k_2k",),
                primary_metric="f1_weighted",
                notes=(
                    "Smallest working configuration to validate the pipeline quickly. "
                    "Expected F1 ~0.85+ on demo slice."
                ),
            )

        if objective == "balanced":
            return AgencyPlan(
                objective="balanced",
                mode="demo",
                feature_family="tfidf+embedding",
                model_family="classical",
                recommended_feature_config="tfidf_uni_bi_5k + sbert/all-MiniLM-L6-v2",
                recommended_model="logistic_regression",
                benchmark_scales=("20k_2k",),   # v2: upgraded from 5k_2k
                primary_metric="f1_weighted",
                notes=(
                    "[v2] Upgraded embedding scale to 20k_2k "
                    "(F1=0.8954 vs 5k_2k F1=0.8736, +2.2%). "
                    "Run best TF-IDF config and one embedding benchmark to compare families."
                ),
            )

        if objective == "best":
            return AgencyPlan(
                objective="best",
                mode="full",
                feature_family="tfidf+embedding",
                model_family="classical",
                recommended_feature_config="tfidf_uni_bi_5k + sbert/all-MiniLM-L6-v2",
                recommended_model="svm",
                benchmark_scales=("5k_2k", "20k_2k"),
                primary_metric="f1_weighted",
                notes=(
                    "Full benchmark coverage, compare both feature families. "
                    "Current best: TF-IDF SVM F1=0.9044 (full, 120k train). "
                    "Finalize report-ready recommendation."
                ),
            )

        raise ValueError(f"Unsupported objective: {objective}")


def plan_to_dict(objective: Objective) -> dict[str, str]:
    return AgencyPlanner().plan(objective).to_dict()
