from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.agency_planner import AgencyPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get an experiment plan for a given objective.")
    parser.add_argument("objective", choices=["fast", "balanced", "best"], help="Experiment objective")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = AgencyPlanner().plan(args.objective)
    pprint(plan.to_dict())


if __name__ == "__main__":
    main()
