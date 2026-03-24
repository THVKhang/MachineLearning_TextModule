import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from modules.tfidf_features import build_tfidf_features, save_features_npy
from modules.bert_embed import EmbedConfig, get_or_build_embeddings
from modules.train_classical import get_model, train_eval
from modules.metrics import print_result 
from modules.text_preprocess import TextCleaner
import os


def build_features(cfg, train_texts, test_texts):
    """
    Preprocess texts (if TF-IDF and preprocessing enabled),
    slice dataset according to demo/full mode,
    and build features (TF-IDF or SBERT embeddings).
    Returns: X_train, X_test, N_train_used, N_test_used
    """

    # ===== Slice dataset according to mode =====
    if cfg.mode == "demo":
        N_train = cfg.n_train_demo
        N_test = cfg.n_test_demo
    elif cfg.mode == "full":
        N_train = len(train_texts)
        N_test = len(test_texts)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    train_texts = train_texts[:N_train]
    test_texts = test_texts[:N_test]

    # ===== Preprocessing for TF-IDF only =====
    if cfg.feature_method == "tfidf" and getattr(cfg, "use_preprocessing", False):
        print("Preprocessing text for TF-IDF...")
        cleaner = TextCleaner(
            remove_stopwords=getattr(cfg, "remove_stopwords", True),
            remove_punctuation=getattr(cfg, "remove_punctuation", True),
            remove_numbers=getattr(cfg, "remove_numbers", True),
        )
        train_texts = cleaner.clean_corpus(train_texts)
        test_texts = cleaner.clean_corpus(test_texts)

    # ===== TF-IDF =====
    if cfg.feature_method == "tfidf":
        cfg.tfidf_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_test, _ = build_tfidf_features(
            train_texts,
            test_texts,
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
        )

        train_path, test_path = save_features_npy(
            X_train,
            X_test,
            feature_dir=str(cfg.tfidf_dir),
            train_name=cfg.tfidf_train_name,
            test_name=cfg.tfidf_test_name,
        )

        print("X_train:", X_train.shape)
        print("X_test:", X_test.shape)
        print("Saved:", train_path)
        print("Saved:", test_path)
        print("Files exist?", os.path.exists(train_path), os.path.exists(test_path))

        return X_train, X_test, N_train, N_test

    # ===== SBERT =====
    elif cfg.feature_method == "sbert":
        if getattr(cfg, "use_preprocessing", False):
            print("Warning: Preprocessing is ignored for SBERT")

        cfg.bert_dir.mkdir(parents=True, exist_ok=True)

        emb_cfg = EmbedConfig(
            model_name=cfg.sbert_model_name,
            batch_size=cfg.sbert_batch_size,
            normalize=cfg.sbert_normalize,
            device=None
        )

        emb_train, emb_test, p_train, p_test = get_or_build_embeddings(
            train_texts,
            test_texts,
            feature_dir=str(cfg.bert_dir),
            train_name=cfg.bert_train_name,
            test_name=cfg.bert_test_name,
            cfg=emb_cfg,
            rebuild=False
        )

        # If cached embeddings were generated with a different slice size,
        # force rebuild so train/test rows always match current mode.
        if emb_train.shape[0] != N_train or emb_test.shape[0] != N_test:
            print(
                "Cached SBERT shape mismatch "
                f"(cached train/test={emb_train.shape[0]}/{emb_test.shape[0]}, "
                f"expected={N_train}/{N_test}) -> rebuilding..."
            )
            emb_train, emb_test, p_train, p_test = get_or_build_embeddings(
                train_texts,
                test_texts,
                feature_dir=str(cfg.bert_dir),
                train_name=cfg.bert_train_name,
                test_name=cfg.bert_test_name,
                cfg=emb_cfg,
                rebuild=True,
            )

        print("SBERT embeddings saved:")
        print(p_train, p_test)

        return emb_train, emb_test, N_train, N_test

    else:
        raise ValueError("Unknown feature_method")


def run_evaluation(X_train, y_train, X_test, y_test, cfg, method_name="Model"):
    """
    Train the model from cfg, evaluate, and display confusion matrix.
    """
    # Slice labels to match features
    y_train = y_train[:len(X_train)]
    y_test = y_test[:len(X_test)]

    # 1. Build model
    model = get_model(
        model_type=cfg.model_type,
        C=getattr(cfg, "C", 1.0),
        max_iter=getattr(cfg, "max_iter", 200),
    )

    # 2. Train + evaluate
    result = train_eval(model, X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)

    metrics = print_result(result)
    metrics["method"] = method_name

    df = pd.DataFrame([metrics])

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {method_name}")
    plt.show()

    return result, df