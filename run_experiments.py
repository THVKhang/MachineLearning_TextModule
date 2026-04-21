from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modules.config import Config
from modules.data_loader import load_data
from modules.eda_advanced import analyze_errors
from modules.metrics import plot_confusion_matrix, print_result
from modules.text_preprocess import TextCleaner
from modules.tfidf_features import build_tfidf_features, save_features_npy
from modules.train_classical import get_model, train_eval


def clean_large_corpus(cleaner: TextCleaner, corpus: list[str], batch_print: int = 20000) -> list[str]:
    cleaned: list[str] = []
    total = len(corpus)
    for i, text in enumerate(corpus):
        cleaned.append(cleaner.clean_text(text))
        if (i + 1) % batch_print == 0:
            print(f"  -> Cleaned {i + 1}/{total} texts...")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified TF-IDF benchmark for 3 classical models.")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo", help="Data mode (default: demo)")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional TF-IDF config names to consider (default: all standard configs)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model list to train (default: config.model_list)",
    )
    return parser.parse_args()


def _default_tfidf_configs() -> list[dict[str, object]]:
    return [
        {"name": "tfidf_uni_1k", "max_features": 1000, "ngram_range": (1, 1), "min_df": 3},
        {"name": "tfidf_uni_bi_3k", "max_features": 3000, "ngram_range": (1, 2), "min_df": 3},
        {"name": "tfidf_uni_bi_5k", "max_features": 5000, "ngram_range": (1, 2), "min_df": 3},
    ]


def run_tfidf_benchmark(
    mode: str = "demo",
    config_names: list[str] | None = None,
    model_list: list[str] | None = None,
    primary_metric: str = "f1_weighted",
) -> dict[str, object]:
    if primary_metric != "f1_weighted":
        raise ValueError("Only f1_weighted is supported as primary_metric in this runner.")

    base_cfg = Config()

    print("=== STEP 1: LOAD DATA ===")
    train_texts, y_train, test_texts, y_test, info = load_data(base_cfg.dataset_name)

    # Use a bounded demo slice by default so one command can finish quickly.
    if mode == "demo":
        n_train = min(base_cfg.n_train_demo, len(train_texts))
        n_test = min(base_cfg.n_test_demo, len(test_texts))
    else:
        n_train = len(train_texts)
        n_test = len(test_texts)

    train_texts = list(train_texts[:n_train])
    test_texts = list(test_texts[:n_test])
    y_train = np.array(y_train[:n_train])
    y_test = np.array(y_test[:n_test])

    print(f"Dataset: {base_cfg.dataset_name}")
    print(f"Mode: {mode}")
    print(f"Train size: {len(train_texts)} / Test size: {len(test_texts)}")
    print(f"Original train/test: {info.train_size}/{info.test_size}\n")

    print("=== STEP 2: PREPROCESS TEXT ===")
    preprocess_start = time.time()
    cleaner = TextCleaner(
        remove_stopwords=base_cfg.remove_stopwords,
        remove_punctuation=base_cfg.remove_punctuation,
        remove_numbers=base_cfg.remove_numbers,
    )
    train_clean = clean_large_corpus(cleaner, train_texts, batch_print=max(1, len(train_texts) // 5))
    test_clean = clean_large_corpus(cleaner, test_texts, batch_print=max(1, len(test_texts) // 5))
    print(f"Preprocessing done in {time.time() - preprocess_start:.2f}s\n")

    print("=== STEP 3: SELECT BEST TF-IDF CONFIG (PRIMARY METRIC: F1-WEIGHTED) ===")
    available_configs = _default_tfidf_configs()
    if config_names is not None:
        allowed = set(config_names)
        tfidf_configs = [cfg for cfg in available_configs if str(cfg["name"]) in allowed]
        if not tfidf_configs:
            raise ValueError("No matching TF-IDF configs found for --configs input.")
    else:
        tfidf_configs = available_configs

    best_cfg: dict[str, object] | None = None
    best_f1 = -1.0

    for tfidf_cfg in tfidf_configs:
        print(f"\n>> Trying {tfidf_cfg['name']}...")
        x_train_tmp, x_test_tmp, _ = build_tfidf_features(
            train_clean,
            test_clean,
            max_features=int(tfidf_cfg["max_features"]),
            ngram_range=tuple(tfidf_cfg["ngram_range"]),
            min_df=int(tfidf_cfg["min_df"]),
        )

        model = get_model("logistic_regression", max_iter=500)
        result = train_eval(
            model=model,
            X_train=x_train_tmp,
            y_train=y_train,
            X_test=x_test_tmp,
            y_test=y_test,
        )
        metrics = print_result(result)
        print(f"   Accuracy={metrics['accuracy']:.4f} | F1-weighted={metrics['f1_weighted']:.4f}")

        if metrics["f1_weighted"] > best_f1:
            best_f1 = float(metrics["f1_weighted"])
            best_cfg = tfidf_cfg

    assert best_cfg is not None
    print(f"\nSelected TF-IDF config: {best_cfg['name']} (F1-weighted={best_f1:.4f})")

    print("\n=== STEP 4: BUILD FINAL TF-IDF FEATURES AND SAVE ===")
    x_train, x_test, _ = build_tfidf_features(
        train_clean,
        test_clean,
        max_features=int(best_cfg["max_features"]),
        ngram_range=tuple(best_cfg["ngram_range"]),
        min_df=int(best_cfg["min_df"]),
    )

    # --- Scale-aware subfolder (avoids demo/full overwrite) ---
    if mode == "demo":
        scale_tag = f"{n_train // 1000}k_{n_test // 1000}k"
    else:
        scale_tag = "full"

    tfidf_dir = base_cfg.feature_dir / "tfidf"
    scale_dir = tfidf_dir / scale_tag
    scale_dir.mkdir(parents=True, exist_ok=True)
    save_features_npy(
        x_train, x_test,
        feature_dir=str(scale_dir),
        train_name="tfidf_train.npy",
        test_name="tfidf_test.npy",
    )
    # Also write metadata so downstream tools know what was built
    (scale_dir / "metadata.json").write_text(
        json.dumps(
            {
                "scale_tag": scale_tag,
                "mode": mode,
                "n_train": n_train,
                "n_test": n_test,
                "tfidf_config": str(best_cfg["name"]),
                "max_features": int(best_cfg["max_features"]),
                "ngram_range": list(best_cfg["ngram_range"]),
                "min_df": int(best_cfg["min_df"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Keep flat files for backward compatibility (overwrite latest)
    tfidf_dir.mkdir(parents=True, exist_ok=True)
    save_features_npy(
        x_train, x_test,
        feature_dir=str(tfidf_dir),
        train_name="tfidf_train.npy",
        test_name="tfidf_test.npy",
    )

    print("\n=== STEP 5: TRAIN 3 MODELS + EXPORT METRICS + CONFUSION MATRICES ===")
    models = model_list if model_list is not None else list(base_cfg.model_list)
    metrics_rows: list[dict[str, float | str]] = []
    artifact_paths: list[str] = []
    eda_paths: list[str] = []  # error analysis JSON per model

    base_cfg.result_dir.mkdir(parents=True, exist_ok=True)
    base_cfg.table_dir.mkdir(parents=True, exist_ok=True)
    base_cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    base_cfg.log_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = base_cfg.result_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # AG News class names (label 0-3)
    class_names = [str(x) for x in sorted(np.unique(y_train))]
    ag_class_names = ["World", "Sports", "Business", "Sci/Tech"]
    # Use AG News names if label set matches
    if len(class_names) == 4:
        class_names_display = ag_class_names
    else:
        class_names_display = class_names

    for model_type in models:
        train_start = time.time()
        model = get_model(model_type, C=base_cfg.C, max_iter=max(500, base_cfg.max_iter))

        result = train_eval(
            model=model,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
        )
        train_time = time.time() - train_start
        y_pred = model.predict(x_test)
        metrics = print_result(result)

        metrics_rows.append(
            {
                "feature_method": "tfidf",
                "tfidf_config": str(best_cfg["name"]),
                "model": model_type,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "train_time_sec": round(train_time, 4),
                "accuracy": round(metrics["accuracy"], 6),
                "precision_weighted": round(metrics["precision_weighted"], 6),
                "recall_weighted": round(metrics["recall_weighted"], 6),
                "f1_weighted": round(metrics["f1_weighted"], 6),
                "primary_metric": primary_metric,
            }
        )

        cm_figure_path = base_cfg.figure_dir / f"cm_{model_type}.png"
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names_display,
            save_path=str(cm_figure_path),
            title=f"Confusion Matrix - {model_type}",
        )
        artifact_paths.append(str(cm_figure_path))

        # Required compatibility location requested in week checkpoints
        cm_root_path = base_cfg.result_dir / f"cm_{model_type}.png"
        shutil.copyfile(cm_figure_path, cm_root_path)
        artifact_paths.append(str(cm_root_path))
        print(f"Saved confusion matrix: {cm_root_path}")

        # Error analysis — save per-model JSON
        error_report = analyze_errors(
            texts=list(test_clean),
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names_display,
            model_name=model_type,
            top_n=10,
        )
        error_path = eda_dir / f"error_analysis_{model_type}.json"
        error_path.write_text(
            json.dumps(error_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        eda_paths.append(str(error_path))
        print(f"Saved error analysis: {error_path}")

    df = pd.DataFrame(metrics_rows).sort_values(by="f1_weighted", ascending=False)
    table_path = base_cfg.table_dir / "tfidf_model_comparison.csv"
    df.to_csv(table_path, index=False)

    print("\nTF-IDF model comparison table:")
    print(df.to_string(index=False))
    print(f"\nSaved table: {table_path}")

    best_row = df.iloc[0].to_dict()
    run_meta: dict[str, object] = {
        "runner": "tfidf",
        "mode": mode,
        "scale_tag": scale_tag,
        "selected_config": str(best_cfg["name"]),
        "candidate_configs": [str(c["name"]) for c in tfidf_configs],
        "models": models,
        "primary_metric": primary_metric,
        "best_model": str(best_row["model"]),
        "best_primary_metric": float(best_row[primary_metric]),
        "table_path": str(table_path),
        "artifact_paths": artifact_paths,
        "eda_error_paths": eda_paths,
    }
    log_path = base_cfg.log_dir / "tfidf_runner_last.json"
    log_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    run_meta["log_path"] = str(log_path)
    return run_meta


def main() -> None:
    args = parse_args()
    run_tfidf_benchmark(
        mode=args.mode,
        config_names=args.configs,
        model_list=args.models,
        primary_metric="f1_weighted",
    )


if __name__ == "__main__":
    main()
