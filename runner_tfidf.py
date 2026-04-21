"""
[DEPRECATED] runner_tfidf.py — Root-level standalone runner.

This file is superseded by the agency workflow pipeline:
  - Use `run_experiments.py::run_tfidf_benchmark()` for programmatic calls.
  - Use `scripts/run_agency_workflow.py <objective>` for the full Planner→Runner→Critic→Reporter workflow.

Kept for reference only. Do NOT add new features here.
"""
import warnings
warnings.warn(
    "runner_tfidf.py is deprecated. Use run_experiments.py or scripts/run_agency_workflow.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import time
import numpy as np
from dataclasses import replace
from pathlib import Path

from modules.config import Config
from modules.data_loader import load_data
from modules.text_preprocess import TextCleaner
from modules.tfidf_features import build_tfidf_features, save_features_npy
from modules.train_classical import get_model, train_eval
from modules.metrics import print_result, plot_confusion_matrix

def run_tfidf_pipeline(cfg: Config, run_name: str):
    """
    Dedicated Runner that receives config from Planner and executes the TF-IDF branch.
    """
    print(f"\nSTARTING TF-IDF RUNNER | Config: {run_name} | Mode: {cfg.mode.upper()}")
    start_time = time.time()

    cfg.tfidf_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    train_texts, y_train, test_texts, y_test, info = load_data(
        cfg.dataset_name, 
        text_field=cfg.text_column, 
        label_field=cfg.label_column
    )
    
    if cfg.mode == "demo":
        train_texts, y_train = train_texts[:cfg.n_train_demo], y_train[:cfg.n_train_demo]
        test_texts, y_test = test_texts[:cfg.n_test_demo], y_test[:cfg.n_test_demo]

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if cfg.use_preprocessing:
        print("[1/3] Cleaning text (Preprocessing)...")
        cleaner = TextCleaner(
            remove_stopwords=cfg.remove_stopwords,
            remove_punctuation=cfg.remove_punctuation,
            remove_numbers=cfg.remove_numbers
        )
        train_texts = cleaner.clean_corpus(train_texts)
        test_texts = cleaner.clean_corpus(test_texts)

    print(f"[2/3] Extracting TF-IDF (Max features: {cfg.max_features}, N-gram: {cfg.ngram_range})...")
    X_train, X_test, vectorizer = build_tfidf_features(
        train_texts, test_texts,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df
    )

    train_path, test_path = save_features_npy(
        X_train, X_test,
        feature_dir=str(cfg.tfidf_dir),
        train_name=f"tfidf_train_{run_name}.npy",
        test_name=f"tfidf_test_{run_name}.npy"
    )

    print(f"[3/3] Training and evaluating {cfg.model_type} model...")
    model = get_model(cfg.model_type, C=cfg.C, max_iter=cfg.max_iter, random_state=cfg.seed)
    result = train_eval(model, X_train, y_train, X_test, y_test)
    metrics_dict = print_result(result)

    class_names = [str(i) for i in range(info.num_classes)] if info.num_classes else ["0", "1", "2", "3"]
    cm_path = cfg.figure_dir / f"cm_{run_name}_{cfg.model_type}.png"
    plot_confusion_matrix(y_test, model.predict(X_test), class_names, str(cm_path), title=f"TF-IDF ({run_name}) - {cfg.model_type}")

    exec_time = time.time() - start_time
    log_file = cfg.log_dir / f"log_{run_name}_{cfg.mode}.json"
    
    log_data = {
        "pipeline": "TF-IDF",
        "run_name": run_name,
        "mode": cfg.mode,
        "execution_time_seconds": round(exec_time, 2),
        "config": {
            "model_type": cfg.model_type,
            "max_features": cfg.max_features,
            "ngram_range": cfg.ngram_range,
            "seed": cfg.seed
        },
        "metrics": metrics_dict,
        "artifacts": {
            "train_npy": str(train_path),
            "test_npy": str(test_path),
            "confusion_matrix": str(cm_path)
        }
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"COMPLETED! F1-Score: {metrics_dict['f1_weighted']:.4f}")
    print(f"Exported Log for Reporter at: {log_file}")
    
    return log_data

if __name__ == "__main__":
    # DEPRECATED: Use `python scripts/run_agency_workflow.py best` instead.
    # Kept for reference. Configs below match historical runs.
    base_config = Config(mode="demo", model_type="logistic_regression")

    config_best = replace(base_config, max_features=10000, ngram_range=(1, 2))
    run_tfidf_pipeline(config_best, run_name="tfidf_best_10k_bigram")

    config_fast = replace(base_config, max_features=3000, ngram_range=(1, 1))
    run_tfidf_pipeline(config_fast, run_name="tfidf_fast_3k_unigram")
