from __future__ import annotations

import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["dataset"]["file"]
PREP = ROOT / "work" / "preprocessing"
OUT = ROOT / "results" / "cv"
OUT.mkdir(parents=True, exist_ok=True)

SEED = int(CONFIG["experiment"]["random_state"])
FOLDS = int(CONFIG["experiment"]["cv_folds"])
THREADS = int(CONFIG["experiment"]["threads"])
TOP_K = int(CONFIG["experiment"]["top_k"])
CORR_THRESHOLD = float(CONFIG["experiment"]["correlation_threshold"])
TARGET = "Label"
DROP_COLUMNS = ["Unnamed: 0", "Attack Type", "Attack Tool"]
CATEGORICAL_CANDIDATES = ["Proto", "sDSb", "dDSb", "Cause", "State"]
BATCH = 20_000

spec = importlib.util.spec_from_file_location("prepare", ROOT / "src" / "01_prepare_data.py")
prepare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prepare)
correlation_reduce = prepare.correlation_reduce
anova_rank = prepare.anova_rank


def metric_row(y_true, prediction, score):
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


def preprocess_fold(data, train_indices, validation_indices, predictors, categorical, numeric):
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)

    Xn_train = numeric_imputer.fit_transform(data.iloc[train_indices][numeric]).astype(np.float32)
    Xn_validation = numeric_imputer.transform(data.iloc[validation_indices][numeric]).astype(np.float32)
    categorical_train = categorical_imputer.fit_transform(data.iloc[train_indices][categorical])
    categorical_validation = categorical_imputer.transform(data.iloc[validation_indices][categorical])
    Xc_train = encoder.fit_transform(categorical_train).astype(np.float32)
    Xc_validation = encoder.transform(categorical_validation).astype(np.float32)

    numeric_position = {column: i for i, column in enumerate(numeric)}
    categorical_position = {column: i for i, column in enumerate(categorical)}

    def assemble(X_numeric, X_categorical):
        return np.ascontiguousarray(
            np.column_stack(
                [
                    X_categorical[:, categorical_position[column]]
                    if column in categorical_position
                    else X_numeric[:, numeric_position[column]]
                    for column in predictors
                ]
            ).astype(np.float32)
        )

    return assemble(Xn_train, Xc_train), assemble(Xn_validation, Xc_validation)


def select_top10(X_train, X_validation, y_train, feature_names, remove_sequence=False):
    variances = np.var(X_train.astype(np.float64), axis=0)
    nonzero = np.isfinite(variances) & (variances > 0)
    usable = [feature_names[i] for i in np.where(nonzero)[0]]
    X_train = X_train[:, nonzero]
    X_validation = X_validation[:, nonzero]

    if remove_sequence:
        kept = [feature for feature in usable if feature not in {"Seq", "Offset"}]
        indices = [usable.index(feature) for feature in kept]
        X_train = X_train[:, indices]
        X_validation = X_validation[:, indices]
        usable = kept

    retained, _, _ = correlation_reduce(X_train, y_train, usable, CORR_THRESHOLD)
    retained_indices = [usable.index(feature) for feature in retained]
    X_train_corr = X_train[:, retained_indices]
    X_validation_corr = X_validation[:, retained_indices]
    ranking = anova_rank(X_train_corr, y_train, retained)
    selected = ranking.head(TOP_K)["Feature"].tolist()
    selected_indices = [retained.index(feature) for feature in selected]
    return (
        np.ascontiguousarray(X_train_corr[:, selected_indices]),
        np.ascontiguousarray(X_validation_corr[:, selected_indices]),
        selected,
        len(usable),
        len(retained),
    )


def run_random_forest(X_train, X_validation, y_train, seed):
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
    prediction = model.predict(X_validation)
    score = model.predict_proba(X_validation)[:, 1]
    prediction_time = time.perf_counter() - started
    return prediction, score, training_time, prediction_time


def run_xgboost(X_train, X_validation, y_train, seed):
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
    score = model.predict_proba(X_validation)[:, 1]
    prediction = (score >= 0.5).astype(np.int8)
    prediction_time = time.perf_counter() - started
    return prediction, score, training_time, prediction_time


def run_rff_svm(X_train, X_validation, y_train, seed):
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
        for start in range(0, len(indices), BATCH):
            batch = indices[start : start + BATCH]
            transformed = transformer.transform(
                scaler.transform(np.asarray(X_train[batch], dtype=np.float32)).astype(np.float32)
            ).astype(np.float32)
            sample_weights = np.where(y_train[batch] == 0, weights[0], weights[1])
            classifier.partial_fit(
                transformed,
                y_train[batch],
                classes=classes if first else None,
                sample_weight=sample_weights,
            )
            first = False
    training_time = time.perf_counter() - started + scaler_time

    predictions = []
    scores = []
    started = time.perf_counter()
    for start in range(0, len(X_validation), BATCH):
        transformed = transformer.transform(
            scaler.transform(np.asarray(X_validation[start : start + BATCH], dtype=np.float32)).astype(np.float32)
        ).astype(np.float32)
        predictions.append(classifier.predict(transformed).astype(np.int8))
        scores.append(classifier.decision_function(transformed).astype(np.float32))
    prediction_time = time.perf_counter() - started
    return np.concatenate(predictions), np.concatenate(scores), training_time, prediction_time


def main() -> None:
    header = pd.read_csv(DATA, nrows=0).columns.tolist()
    predictors = [column for column in header if column not in DROP_COLUMNS + [TARGET]]
    categorical = [column for column in predictors if column in CATEGORICAL_CANDIDATES]
    numeric = [column for column in predictors if column not in categorical]

    data = pd.read_csv(DATA, usecols=predictors + [TARGET], low_memory=False)
    duplicate_positions = pd.read_csv(PREP / "duplicate_rows_removed.csv")["duplicate_row_position"].to_numpy(dtype=np.int64)
    keep = np.ones(len(data), dtype=bool)
    keep[duplicate_positions] = False
    data = data.loc[keep].reset_index(drop=True)
    del keep

    y = (data[TARGET].astype(str).str.strip().str.lower() == "malicious").astype(np.int8).to_numpy()
    development_indices = np.load(PREP / "train_indices.npy")
    development = data.iloc[development_indices].reset_index(drop=True)
    y_development = y[development_indices]

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    result_rows = []
    selected_rows = []
    preprocessing_rows = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(y_development)), y_development), start=1
    ):
        seed = SEED + fold
        started = time.perf_counter()
        X_train_raw, X_validation_raw = preprocess_fold(
            development,
            train_indices,
            validation_indices,
            predictors,
            categorical,
            numeric,
        )
        base_preprocessing_time = time.perf_counter() - started

        for condition, remove_sequence in [
            ("Top-10", False),
            ("Top-10 without Seq/Offset", True),
        ]:
            started = time.perf_counter()
            X_train, X_validation, selected, usable_count, retained_count = select_top10(
                X_train_raw,
                X_validation_raw,
                y_development[train_indices],
                predictors,
                remove_sequence,
            )
            selection_time = time.perf_counter() - started
            preprocessing_rows.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "base_preprocessing_time_s": base_preprocessing_time,
                    "feature_selection_time_s": selection_time,
                    "usable_features": usable_count,
                    "correlation_reduced_features": retained_count,
                    "top_features": "|".join(selected),
                }
            )
            selected_rows.extend(
                {
                    "fold": fold,
                    "condition": condition,
                    "rank": rank,
                    "feature": feature,
                }
                for rank, feature in enumerate(selected, start=1)
            )

            runners = [
                ("Random Forest", run_random_forest),
                ("XGBoost", run_xgboost),
                ("RFF-SVM", run_rff_svm),
            ]
            for model_name, runner in runners:
                prediction, score, training_time, prediction_time = runner(
                    X_train,
                    X_validation,
                    y_development[train_indices],
                    seed,
                )
                row = {
                    "fold": fold,
                    "condition": condition,
                    "model": model_name,
                    "seed": seed,
                    "n_features": X_train.shape[1],
                    "n_train": len(train_indices),
                    "n_validation": len(validation_indices),
                    "training_time_s": training_time,
                    "prediction_time_s": prediction_time,
                    "selected_features": "|".join(selected),
                }
                row.update(metric_row(y_development[validation_indices], prediction, score))
                result_rows.append(row)
                pd.DataFrame(result_rows).to_csv(OUT / "cv_fold_results_partial.csv", index=False)
                del prediction, score
                gc.collect()

            del X_train, X_validation
            gc.collect()
        del X_train_raw, X_validation_raw
        gc.collect()

    fold_results = pd.DataFrame(result_rows)
    fold_results.to_csv(OUT / "cv_fold_results.csv", index=False)
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "fnr",
        "roc_auc",
        "pr_auc",
        "mcc",
        "training_time_s",
        "prediction_time_s",
    ]
    summary = fold_results.groupby(["condition", "model"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(value) for value in column if value) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(OUT / "cv_summary.csv", index=False)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(OUT / "cv_selected_features.csv", index=False)
    pd.DataFrame(preprocessing_rows).to_csv(OUT / "cv_preprocessing_times.csv", index=False)
    stability = selected_df.groupby(["condition", "feature"]).size().reset_index(name="selection_count")
    stability["selection_frequency"] = stability["selection_count"] / FOLDS
    stability.sort_values(
        ["condition", "selection_count", "feature"], ascending=[True, False, True]
    ).to_csv(OUT / "cv_feature_stability.csv", index=False)


if __name__ == "__main__":
    main()
