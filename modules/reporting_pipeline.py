"""
Automated reporting pipeline for ML experiments.
Aggregates configs, metrics, predictions, and artifacts.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from modules.config import Config
from modules.metrics import EvalResult

class ExperimentTracker:
    """Stores and manages experiment runs."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.storage_dir / "experiments.json"
        self._load_runs()
    
    def _load_runs(self):
        if self.runs_file.exists():
            with open(self.runs_file, 'r') as f:
                self.runs = json.load(f)
        else:
            self.runs = []
    
    def _save_runs(self):
        with open(self.runs_file, 'w') as f:
            json.dump(self.runs, f, indent=2, default=str)
    
    def log_run(self, config: Config, metrics: EvalResult, 
                predictions: Optional[np.ndarray] = None,
                artifact_paths: Optional[Dict[str, str]] = None,
                additional_metadata: Optional[Dict] = None):
        """Log a single experiment run."""
        run_id = len(self.runs) + 1
        run_entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(config),
            "metrics": asdict(metrics),
            "artifact_paths": artifact_paths or {},
            "metadata": additional_metadata or {}
        }
        # Optionally save predictions separately (too large for JSON)
        if predictions is not None:
            pred_path = self.storage_dir / f"predictions_run_{run_id}.npy"
            np.save(pred_path, predictions)
            run_entry["artifact_paths"]["predictions"] = str(pred_path)
        
        self.runs.append(run_entry)
        self._save_runs()
    
    def get_runs_as_dataframe(self) -> pd.DataFrame:
        """Return all runs as a DataFrame for easy comparison."""
        rows = []
        for run in self.runs:
            row = {
                "run_id": run["run_id"],
                "timestamp": run["timestamp"],
                **run["metrics"],
                **{f"cfg_{k}": v for k, v in run["config"].items() if isinstance(v, (str, int, float, bool))}
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    def compare_runs(self, metric: str = "f1_weighted", ascending: bool = False) -> pd.DataFrame:
        """Sort runs by a given metric."""
        df = self.get_runs_as_dataframe()
        if metric in df.columns:
            df = df.sort_values(by=metric, ascending=ascending)
        return df
    
    def get_best_run(self, metric: str = "f1_weighted") -> Dict:
        """Return the best run based on a metric."""
        df = self.compare_runs(metric)
        if df.empty:
            return {}
        best_id = df.iloc[0]["run_id"]
        return next(run for run in self.runs if run["run_id"] == best_id)