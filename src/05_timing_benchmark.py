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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / "work" / "preprocessing"
OUT = ROOT / "results" / "timing"
OUT.mkdir(parents=True, exist_ok=True)
THREADS = int(CONFIG["experiment"]["threads"])
REPEATS = int(CONFIG["experiment"]["timing_repeats"])

X_train = np.load(DATA / "X_train_top10.npy", mmap_mode="r")
X_test = np.load(DATA / "X_test_top10.npy", mmap_mode="r")
y_train = np.load(DATA / "y_train.npy")
y_test = np.load(DATA / "y_test.npy")


def random_forest(seed):
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=seed,
        n_jobs=THREADS,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    model.predict(X_test)
    prediction_time = time.perf_counter() - started
    return training_time, prediction_time


def xgboost_model(seed):
    scale_positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_positive_weight,
        random_state=seed,
        n_jobs=THREADS,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    model.predict(X_test)
    prediction_time = time.perf_counter() - started
    return training_time, prediction_time


def rff_svm(seed):
    batch = 20_000
    scaler = StandardScaler()
    started = time.perf_counter()
    scaler.fit(X_train)
    scaler_time = time.perf_counter() - started
    transformer = RBFSampler(gamma=1.0 / X_train.shape[1], n_components=300, random_state=seed)
    transformer.fit(scaler.transform(np.asarray(X_train[:100], dtype=np.float32)))
    classifier = SGDClassifier(
        loss="hinge",
        alpha=1e-5,
        penalty="l2",
        max_iter=1,
        tol=None,
        random_state=seed,
        learning_rate="optimal",
        average=True,
    )
    classes = np.array([0, 1], dtype=np.int8)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train), dtype=np.int64)
    first = True

    started = time.perf_counter()
    for _ in range(5):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch):
            current = indices[start : start + batch]
            transformed = transformer.transform(
                scaler.transform(np.asarray(X_train[current], dtype=np.float32)).astype(np.float32)
            ).astype(np.float32)
            sample_weights = np.where(y_train[current] == 0, weights[0], weights[1])
            classifier.partial_fit(
                transformed,
                y_train[current],
                classes=classes if first else None,
                sample_weight=sample_weights,
            )
            first = False
    training_time = time.perf_counter() - started + scaler_time

    started = time.perf_counter()
    for start in range(0, len(y_test), batch):
        transformed = transformer.transform(
            scaler.transform(np.asarray(X_test[start : start + batch], dtype=np.float32)).astype(np.float32)
        ).astype(np.float32)
        classifier.predict(transformed)
    prediction_time = time.perf_counter() - started
    return training_time, prediction_time


def main() -> None:
    rows = []
    with threadpool_limits(limits=THREADS):
        for repeat in range(1, REPEATS + 1):
            seed = 41 + repeat
            for model_name, runner in [
                ("Random Forest", random_forest),
                ("XGBoost", xgboost_model),
                ("RFF-SVM", rff_svm),
            ]:
                training_time, prediction_time = runner(seed)
                row = {
                    "model": model_name,
                    "repeat": repeat,
                    "seed": seed,
                    "threads": THREADS,
                    "n_features": 10,
                    "training_time_s": training_time,
                    "prediction_time_s": prediction_time,
                    "latency_ms_per_1000": prediction_time / len(y_test) * 1_000_000,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT / "fair_timing_partial.csv", index=False)
                print(row)
                gc.collect()

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "fair_timing_results.csv", index=False)
    results.groupby("model")[["training_time_s", "prediction_time_s", "latency_ms_per_1000"]].agg(
        ["mean", "std"]
    ).to_csv(OUT / "fair_timing_summary.csv")


if __name__ == "__main__":
    main()
