import numpy as np
from dataclasses import replace

from modules.metrics import calculate_metrics
from modules.config import Config
from modules.pipeline import build_features
from modules.bert_embed import get_or_build_embeddings
from modules.tfidf_features import build_tfidf_features, load_features_npy, save_features_npy


def test_calculate_metrics_returns_valid_scores():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    result = calculate_metrics(y_true, y_pred)

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision_weighted <= 1.0
    assert 0.0 <= result.recall_weighted <= 1.0
    assert 0.0 <= result.f1_weighted <= 1.0


def test_build_tfidf_features_shapes_are_consistent():
    train_texts = ["machine learning", "deep learning"]
    test_texts = ["machine model"]

    x_train, x_test, _ = build_tfidf_features(
        train_texts,
        test_texts,
        max_features=10,
        ngram_range=(1, 1),
        min_df=1,
    )

    assert x_train.shape[0] == 2
    assert x_test.shape[0] == 1
    assert x_train.shape[1] == x_test.shape[1]
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32


def test_save_and_load_tfidf_features_roundtrip(tmp_path):
    x_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x_test = np.array([[0.5, 0.5]], dtype=np.float32)

    save_features_npy(
        x_train,
        x_test,
        feature_dir=str(tmp_path),
        train_name="tfidf_train.npy",
        test_name="tfidf_test.npy",
    )

    loaded_train, loaded_test = load_features_npy(
        feature_dir=str(tmp_path),
        train_name="tfidf_train.npy",
        test_name="tfidf_test.npy",
    )

    assert np.array_equal(loaded_train, x_train)
    assert np.array_equal(loaded_test, x_test)


def test_get_or_build_embeddings_loads_existing_cache(tmp_path):
    x_train = np.random.rand(3, 4).astype(np.float32)
    x_test = np.random.rand(2, 4).astype(np.float32)

    save_features_npy(
        x_train,
        x_test,
        feature_dir=str(tmp_path),
        train_name="bert_train.npy",
        test_name="bert_test.npy",
    )

    emb_train, emb_test, p_train, p_test = get_or_build_embeddings(
        ["a", "b", "c"],
        ["x", "y"],
        feature_dir=str(tmp_path),
        train_name="bert_train.npy",
        test_name="bert_test.npy",
        rebuild=False,
    )

    assert p_train.endswith("bert_train.npy")
    assert p_test.endswith("bert_test.npy")
    assert emb_train.shape == (3, 4)
    assert emb_test.shape == (2, 4)


def test_pipeline_sbert_rebuilds_when_cached_shape_mismatch(monkeypatch):
    cfg = replace(
        Config(),
        mode="demo",
        feature_method="sbert",
        n_train_demo=3,
        n_test_demo=2,
    )

    calls = []

    def fake_get_or_build_embeddings(*args, **kwargs):
        calls.append(kwargs.get("rebuild", False))
        if kwargs.get("rebuild", False):
            return (
                np.zeros((3, 8), dtype=np.float32),
                np.zeros((2, 8), dtype=np.float32),
                "train.npy",
                "test.npy",
            )
        return (
            np.zeros((5, 8), dtype=np.float32),
            np.zeros((4, 8), dtype=np.float32),
            "train.npy",
            "test.npy",
        )

    monkeypatch.setattr("modules.pipeline.get_or_build_embeddings", fake_get_or_build_embeddings)

    train_texts = ["t1", "t2", "t3", "t4"]
    test_texts = ["u1", "u2", "u3"]

    x_train, x_test, n_train, n_test = build_features(cfg, train_texts, test_texts)

    assert n_train == 3
    assert n_test == 2
    assert x_train.shape[0] == 3
    assert x_test.shape[0] == 2
    assert calls == [False, True]