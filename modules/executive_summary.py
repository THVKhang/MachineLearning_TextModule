"""
Generates a business‑friendly executive summary from experiment results.
"""

import pandas as pd
from typing import Dict, Any

def generate_executive_summary(experiments_df: pd.DataFrame, 
                               best_metric: str = "f1_weighted",
                               model_names_map: Dict[str, str] = None) -> str:
    """
    Create an automated executive summary.
    
    Args:
        experiments_df: DataFrame from ExperimentTracker.get_runs_as_dataframe()
        best_metric: Primary metric for ranking (e.g., 'f1_weighted', 'accuracy')
        model_names_map: Optional mapping from internal model names to display names.
    
    Returns:
        Markdown string ready for report.
    """
    if experiments_df.empty:
        return "No experiments found."
    
    # Identify best run
    best_run = experiments_df.sort_values(by=best_metric, ascending=False).iloc[0]
    
    # Map model name
    model_type = best_run.get("cfg_model_type", "unknown")
    if model_names_map:
        model_type = model_names_map.get(model_type, model_type)
    
    feature_method = best_run.get("cfg_feature_method", "unknown")
    mode = best_run.get("cfg_mode", "unknown")
    
    summary = f"""
# Executive Summary – Text Classification on AG News

## Best Performing Model
- **Model:** {model_type}  
- **Feature extraction:** {feature_method}  
- **Mode:** {mode}  
- **Primary metric ({best_metric})**: {best_run[best_metric]:.4f}  
- **Accuracy**: {best_run['accuracy']:.4f}  
- **Precision (weighted)**: {best_run['precision_weighted']:.4f}  
- **Recall (weighted)**: {best_run['recall_weighted']:.4f}
"""
    return summary

def save_summary_to_markdown(summary_text: str, output_path: str):
    """Save the summary as a markdown file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)