from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


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

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["benchmark_scales"] = ", ".join(self.benchmark_scales)
        return data


class AgencyPlanner:
    """Simple planner that selects an experiment strategy by objective.

    This is intentionally lightweight: it recommends what to run rather than
    executing anything itself.
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
                notes="Use the smallest working configuration to validate the pipeline quickly.",
            )

        if objective == "balanced":
            return AgencyPlan(
                objective="balanced",
                mode="demo",
                feature_family="tfidf+embedding",
                model_family="classical",
                recommended_feature_config="tfidf_uni_bi_5k + sbert/all-MiniLM-L6-v2",
                recommended_model="logistic_regression",
                benchmark_scales=("5k_2k",),
                primary_metric="f1_weighted",
                notes="Run the best TF-IDF config and one embedding benchmark to compare families.",
            )

        if objective == "best":
            return AgencyPlan(
                objective="best",
                mode="full",
                feature_family="tfidf+embedding",
                model_family="classical",
                recommended_feature_config="tfidf_uni_bi_5k + sbert/all-MiniLM-L6-v2",
                recommended_model="logistic_regression",
                benchmark_scales=("5k_2k", "20k_2k"),
                primary_metric="f1_weighted",
                notes="Run full benchmark coverage, compare both feature families, and finalize the report-ready recommendation.",
            )

        raise ValueError(f"Unsupported objective: {objective}")


def plan_to_dict(objective: Objective) -> dict[str, str]:
    return AgencyPlanner().plan(objective).to_dict()
