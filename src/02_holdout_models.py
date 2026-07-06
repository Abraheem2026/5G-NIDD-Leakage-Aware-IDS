from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / "work" / "preprocessing"
OUT = ROOT / "results" / "holdout"
OUT.mkdir(parents=True, exist_ok=True)

SEED = int(CONFIG["experiment"]["random_state"])
THREADS = int(CONFIG["experiment"]["threads"])
BATCH = 20_000
RFF_COMPONENTS = 300
RFF_EPOCHS = 5

CONDITIONS = [
    ("Full usable", "full"),
    ("Correlation-reduced", "corr"),
    ("Top-10", "top10"),
    ("Full usable without Seq/Offset", "full_no_seq_offset"),
    ("Correlation-reduced without Seq/Offset", "corr_no_seq_offset"),
    ("Top-10 without Seq/Offset", "top10_no_seq_offset"),
]


def metrics(y_true: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "fpr": float(fp / (tn + fp)),
        "fnr": float(fn / (fn + tp)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_condition(key: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.load(DATA / f"X_train_{key}.npy", mmap_mode="r"),
        np.load(DATA / f"X_test_{key}.npy", mmap_mode="r"),
    )


def random_forest(X_train, X_test, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=THREADS,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    prediction_time = time.perf_counter() - started
    return prediction, score, training_time, prediction_time


def xgboost_model(X_train, X_test, y_train):
    scale_positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_positive_weight,
        random_state=SEED,
        n_jobs=THREADS,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    score = model.predict_proba(X_test)[:, 1]
    prediction = (score >= 0.5).astype(np.int8)
    prediction_time = time.perf_counter() - started
    return prediction, score, training_time, prediction_time


def rff_svm(X_train, X_test, y_train):
    scaler = StandardScaler()
    started = time.perf_counter()
    scaler.fit(X_train)
    scaler_time = time.perf_counter() - started

    transformer = RBFSampler(
        gamma=1.0 / X_train.shape[1],
        n_components=RFF_COMPONENTS,
        random_state=SEED,
    )
    transformer.fit(scaler.transform(np.asarray(X_train[:100], dtype=np.float32)))

    classifier = SGDClassifier(
        loss="hinge",
        alpha=1e-5,
        penalty="l2",
        max_iter=1,
        tol=None,
        random_state=SEED,
        learning_rate="optimal",
        average=True,
    )
    classes = np.array([0, 1], dtype=np.int8)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(y_train), dtype=np.int64)
    first_batch = True

    started = time.perf_counter()
    for _ in range(RFF_EPOCHS):
        rng.shuffle(indices)
        for start in range(0, len(indices), BATCH):
            batch = indices[start : start + BATCH]
            standardized = scaler.transform(np.asarray(X_train[batch], dtype=np.float32)).astype(np.float32)
            transformed = transformer.transform(standardized).astype(np.float32)
            sample_weights = np.where(y_train[batch] == 0, class_weights[0], class_weights[1])
            classifier.partial_fit(
                transformed,
                y_train[batch],
                classes=classes if first_batch else None,
                sample_weight=sample_weights,
            )
            first_batch = False
    training_time = time.perf_counter() - started + scaler_time

    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(X_test), BATCH):
        standardized = scaler.transform(np.asarray(X_test[start : start + BATCH], dtype=np.float32)).astype(np.float32)
        transformed = transformer.transform(standardized).astype(np.float32)
        predictions.append(classifier.predict(transformed).astype(np.int8))
        scores.append(classifier.decision_function(transformed).astype(np.float32))
    prediction_time = time.perf_counter() - started
    return np.concatenate(predictions), np.concatenate(scores), training_time, prediction_time


def main() -> None:
    y_train = np.load(DATA / "y_train.npy")
    y_test = np.load(DATA / "y_test.npy")
    rows: list[dict[str, object]] = []

    runners = [
        ("Random Forest", random_forest, "rf"),
        ("XGBoost", xgboost_model, "xgb"),
        ("RFF-SVM", rff_svm, "rff"),
    ]

    for condition_label, condition_key in CONDITIONS:
        X_train, X_test = load_condition(condition_key)
        for model_name, runner, slug in runners:
            prediction, score, training_time, prediction_time = runner(X_train, X_test, y_train)
            row = {
                "model": model_name,
                "feature_condition": condition_label,
                "feature_key": condition_key,
                "n_features": int(X_train.shape[1]),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "training_time_s": training_time,
                "prediction_time_s": prediction_time,
                "random_state": SEED,
            }
            row.update(metrics(y_test, prediction, score))
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "holdout_results_partial.csv", index=False)

            if condition_key in {"top10", "top10_no_seq_offset"}:
                np.save(OUT / f"pred_{slug}_{condition_key}.npy", prediction.astype(np.int8))
                np.save(OUT / f"score_{slug}_{condition_key}.npy", score.astype(np.float32))
            print(json.dumps(row, indent=2))
            del prediction, score
            gc.collect()
        del X_train, X_test
        gc.collect()

    pd.DataFrame(rows).to_csv(OUT / "holdout_results.csv", index=False)


if __name__ == "__main__":
    main()
