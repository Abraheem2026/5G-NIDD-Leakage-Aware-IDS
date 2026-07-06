from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["dataset"]["file"]
OUT = ROOT / "work" / "preprocessing"
OUT.mkdir(parents=True, exist_ok=True)
IDENTITY_FILE = OUT / "dataset_identity.json"

SEED = int(CONFIG["experiment"]["random_state"])
TEST_SIZE = float(CONFIG["experiment"]["test_size"])
CORR_THRESHOLD = float(CONFIG["experiment"]["correlation_threshold"])
TOP_K = int(CONFIG["experiment"]["top_k"])
CHUNK = 50_000
TARGET = "Label"
DROP_COLUMNS = ["Unnamed: 0", "Attack Type", "Attack Tool"]
CATEGORICAL_CANDIDATES = ["Proto", "sDSb", "dDSb", "Cause", "State"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_duplicate_positions(path: Path, columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Identify exact duplicates using row hashes followed by exact comparison."""
    total = sum(1 for _ in path.open("rb")) - 1
    hash_path = OUT / "_row_hashes.uint64"
    hashes = np.memmap(hash_path, mode="w+", dtype=np.uint64, shape=(total,))
    position = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK, low_memory=False):
        values = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64)
        hashes[position : position + len(values)] = values
        position += len(values)
    hashes.flush()

    unique_hashes, counts = np.unique(np.asarray(hashes), return_counts=True)
    repeated = set(unique_hashes[counts > 1].tolist())
    del unique_hashes, counts, hashes

    candidates: list[pd.DataFrame] = []
    base = 0
    if repeated:
        for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK, low_memory=False):
            values = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64)
            mask = np.fromiter((int(value) in repeated for value in values), dtype=bool, count=len(values))
            if mask.any():
                candidate = chunk.loc[mask].copy()
                candidate.insert(0, "_row_position", np.arange(base, base + len(chunk), dtype=np.int64)[mask])
                candidate.insert(1, "_hash", values[mask])
                candidates.append(candidate)
            base += len(chunk)

    hash_path.unlink(missing_ok=True)
    if not candidates:
        return np.array([], dtype=np.int64), pd.DataFrame()

    candidate_df = pd.concat(candidates, ignore_index=True).sort_values("_row_position").reset_index(drop=True)
    data_columns = [column for column in candidate_df.columns if column not in {"_row_position", "_hash"}]
    duplicate_mask = candidate_df.duplicated(subset=data_columns, keep="first")
    duplicate_positions = candidate_df.loc[duplicate_mask, "_row_position"].to_numpy(dtype=np.int64)
    duplicate_groups = candidate_df[candidate_df.duplicated(subset=data_columns, keep=False)].copy()
    return duplicate_positions, duplicate_groups


def correlation_reduce(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Deterministic, training-only Pearson correlation reduction."""
    n_rows, n_features = X.shape
    sums = np.zeros(n_features, dtype=np.float64)
    cross = np.zeros((n_features, n_features), dtype=np.float64)
    sum_xy = np.zeros(n_features, dtype=np.float64)
    y_float = np.asarray(y, dtype=np.float64)
    y_sum = float(y_float.sum())
    y_sq_sum = float(np.square(y_float).sum())

    for start in range(0, n_rows, CHUNK):
        xb = np.asarray(X[start : start + CHUNK], dtype=np.float64)
        yb = y_float[start : start + CHUNK]
        sums += xb.sum(axis=0)
        cross += xb.T @ xb
        sum_xy += xb.T @ yb

    covariance = (cross - np.outer(sums, sums) / n_rows) / (n_rows - 1)
    variances = np.maximum(np.diag(covariance), 0.0)
    std = np.sqrt(variances)
    denominator = np.outer(std, std)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    np.fill_diagonal(correlation, 1.0)

    y_variance = (y_sq_sum - y_sum * y_sum / n_rows) / (n_rows - 1)
    covariance_xy = (sum_xy - sums * y_sum / n_rows) / (n_rows - 1)
    target_correlation = np.abs(
        np.divide(
            covariance_xy,
            std * np.sqrt(max(y_variance, 0.0)),
            out=np.zeros_like(covariance_xy),
            where=std > 0,
        )
    )

    dropped: set[str] = set()
    pair_rows: list[dict[str, object]] = []
    for i, feature_a in enumerate(feature_names):
        if feature_a in dropped:
            continue
        for j in range(i + 1, len(feature_names)):
            feature_b = feature_names[j]
            if feature_b in dropped or abs(correlation[i, j]) <= threshold:
                continue
            if target_correlation[j] > target_correlation[i]:
                retained, removed = feature_b, feature_a
            else:
                retained, removed = feature_a, feature_b
            dropped.add(removed)
            pair_rows.append(
                {
                    "Feature_A": feature_a,
                    "Feature_B": feature_b,
                    "Pearson_r": float(correlation[i, j]),
                    "Abs_target_corr_A": float(target_correlation[i]),
                    "Abs_target_corr_B": float(target_correlation[j]),
                    "Retained": retained,
                    "Dropped": removed,
                }
            )
            if removed == feature_a:
                break

    retained_features = [feature for feature in feature_names if feature not in dropped]
    correlation_df = pd.DataFrame(correlation, index=feature_names, columns=feature_names)
    pairs_df = pd.DataFrame(pair_rows)
    return retained_features, correlation_df, pairs_df


def anova_rank(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Compute binary one-way ANOVA F scores from sufficient statistics."""
    n_rows, n_features = X.shape
    count_0 = int((y == 0).sum())
    count_1 = int((y == 1).sum())
    sum_0 = np.zeros(n_features, dtype=np.float64)
    sum_1 = np.zeros(n_features, dtype=np.float64)
    sq_0 = np.zeros(n_features, dtype=np.float64)
    sq_1 = np.zeros(n_features, dtype=np.float64)

    for start in range(0, n_rows, CHUNK):
        xb = np.asarray(X[start : start + CHUNK], dtype=np.float64)
        yb = y[start : start + CHUNK]
        if np.any(yb == 0):
            x0 = xb[yb == 0]
            sum_0 += x0.sum(axis=0)
            sq_0 += np.square(x0).sum(axis=0)
        if np.any(yb == 1):
            x1 = xb[yb == 1]
            sum_1 += x1.sum(axis=0)
            sq_1 += np.square(x1).sum(axis=0)

    mean_0 = sum_0 / count_0
    mean_1 = sum_1 / count_1
    grand_mean = (sum_0 + sum_1) / n_rows
    ss_between = count_0 * np.square(mean_0 - grand_mean) + count_1 * np.square(mean_1 - grand_mean)
    ss_within = (sq_0 - count_0 * np.square(mean_0)) + (sq_1 - count_1 * np.square(mean_1))
    denominator = ss_within / max(n_rows - 2, 1)
    f_scores = np.divide(ss_between, denominator, out=np.full(n_features, np.nan), where=denominator > 0)
    p_values = f_distribution.sf(f_scores, 1, n_rows - 2)
    return (
        pd.DataFrame({"Feature": feature_names, "F_Score": f_scores, "p_value": p_values})
        .sort_values("F_Score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def save_array(name: str, array: np.ndarray) -> None:
    np.save(OUT / f"{name}.npy", np.ascontiguousarray(array))


def main() -> None:
    started = time.perf_counter()
    current_hash = sha256_file(DATA)
    expected_hash = CONFIG["dataset"]["expected_sha256"]
    if current_hash != expected_hash:
        raise SystemExit(
            "Dataset SHA-256 does not match the audited manuscript version. "
            "Run check_dataset.py for details."
        )

    header = pd.read_csv(DATA, nrows=0).columns.tolist()
    duplicate_basis = [column for column in header if column != "Unnamed: 0"]

    duplicate_file = OUT / "duplicate_rows_removed.csv"
    cached_identity = {}
    if IDENTITY_FILE.exists():
        cached_identity = json.loads(IDENTITY_FILE.read_text(encoding="utf-8"))
    cache_matches = (
        duplicate_file.exists()
        and cached_identity.get("sha256") == current_hash
        and cached_identity.get("columns") == len(header)
    )

    if cache_matches:
        duplicate_positions = pd.read_csv(duplicate_file)["duplicate_row_position"].to_numpy(dtype=np.int64)
        duplicate_groups = pd.DataFrame()
        print("Reusing duplicate audit cached for the verified dataset hash.")
    else:
        duplicate_positions, duplicate_groups = exact_duplicate_positions(DATA, duplicate_basis)
        pd.DataFrame({"duplicate_row_position": duplicate_positions}).to_csv(duplicate_file, index=False)
        if not duplicate_groups.empty:
            duplicate_groups.to_csv(OUT / "duplicate_groups_exact.csv", index=False)
        IDENTITY_FILE.write_text(
            json.dumps({"sha256": current_hash, "columns": len(header)}, indent=2),
            encoding="utf-8",
        )

    predictor_names = [column for column in header if column not in DROP_COLUMNS + [TARGET]]
    categorical = [column for column in predictor_names if column in CATEGORICAL_CANDIDATES]
    numeric = [column for column in predictor_names if column not in categorical]

    data = pd.read_csv(DATA, usecols=predictor_names + [TARGET], low_memory=False)
    audit_labels = pd.read_csv(DATA, usecols=[TARGET, "Attack Type"], low_memory=False)
    raw_records = len(data)

    duplicate_summary = (
        audit_labels.iloc[duplicate_positions]
        .groupby([TARGET, "Attack Type"], dropna=False)
        .size()
        .reset_index(name="removed_records")
        if len(duplicate_positions)
        else pd.DataFrame(columns=[TARGET, "Attack Type", "removed_records"])
    )
    duplicate_summary.to_csv(OUT / "duplicate_class_summary.csv", index=False)

    if len(duplicate_positions):
        keep = np.ones(raw_records, dtype=bool)
        keep[duplicate_positions] = False
        data = data.loc[keep].reset_index(drop=True)
        audit_labels = audit_labels.loc[keep].reset_index(drop=True)
        del keep

    class_distribution = (
        audit_labels.groupby([TARGET, "Attack Type"], dropna=False)
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
        .reset_index(drop=True)
    )
    class_distribution.to_csv(OUT / "class_distribution_after_deduplication.csv", index=False)
    attack_type_counts = {
        str(row["Attack Type"]): int(row["records"])
        for _, row in class_distribution.iterrows()
    }
    del audit_labels
    gc.collect()

    y = (data[TARGET].astype(str).str.strip().str.lower() == "malicious").astype(np.int8).to_numpy()
    indices = np.arange(len(data), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    y_train = y[train_indices]
    y_test = y[test_indices]

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)

    Xn_train = numeric_imputer.fit_transform(data.iloc[train_indices][numeric]).astype(np.float32)
    Xn_test = numeric_imputer.transform(data.iloc[test_indices][numeric]).astype(np.float32)
    categorical_train = categorical_imputer.fit_transform(data.iloc[train_indices][categorical])
    categorical_test = categorical_imputer.transform(data.iloc[test_indices][categorical])
    Xc_train = encoder.fit_transform(categorical_train).astype(np.float32)
    Xc_test = encoder.transform(categorical_test).astype(np.float32)

    numeric_position = {column: i for i, column in enumerate(numeric)}
    categorical_position = {column: i for i, column in enumerate(categorical)}

    def assemble(X_numeric: np.ndarray, X_categorical: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack(
                [
                    X_categorical[:, categorical_position[column]]
                    if column in categorical_position
                    else X_numeric[:, numeric_position[column]]
                    for column in predictor_names
                ]
            ).astype(np.float32)
        )

    X_train_raw = assemble(Xn_train, Xc_train)
    X_test_raw = assemble(Xn_test, Xc_test)
    del Xn_train, Xn_test, Xc_train, Xc_test, categorical_train, categorical_test
    gc.collect()

    variances = np.var(X_train_raw.astype(np.float64), axis=0)
    nonzero = np.isfinite(variances) & (variances > 0)
    zero_variance_features = [predictor_names[i] for i in np.where(~nonzero)[0]]
    usable_features = [predictor_names[i] for i in np.where(nonzero)[0]]
    X_train_full = np.ascontiguousarray(X_train_raw[:, nonzero])
    X_test_full = np.ascontiguousarray(X_test_raw[:, nonzero])
    del X_train_raw, X_test_raw
    gc.collect()

    pd.DataFrame(
        {"Feature": predictor_names, "Training_variance": variances, "Retained": nonzero}
    ).to_csv(OUT / "training_variance_filter.csv", index=False)

    retained, correlation, high_pairs = correlation_reduce(
        X_train_full, y_train, usable_features, CORR_THRESHOLD
    )
    retained_indices = [usable_features.index(feature) for feature in retained]
    X_train_corr = np.ascontiguousarray(X_train_full[:, retained_indices])
    X_test_corr = np.ascontiguousarray(X_test_full[:, retained_indices])
    ranking = anova_rank(X_train_corr, y_train, retained)
    top10_features = ranking.head(TOP_K)["Feature"].tolist()
    top10_indices = [retained.index(feature) for feature in top10_features]
    X_train_top10 = np.ascontiguousarray(X_train_corr[:, top10_indices])
    X_test_top10 = np.ascontiguousarray(X_test_corr[:, top10_indices])

    no_sequence_features = [feature for feature in usable_features if feature not in {"Seq", "Offset"}]
    no_sequence_indices = [usable_features.index(feature) for feature in no_sequence_features]
    X_train_no_sequence_full = np.ascontiguousarray(X_train_full[:, no_sequence_indices])
    X_test_no_sequence_full = np.ascontiguousarray(X_test_full[:, no_sequence_indices])
    retained_no_sequence, correlation_no_sequence, pairs_no_sequence = correlation_reduce(
        X_train_no_sequence_full,
        y_train,
        no_sequence_features,
        CORR_THRESHOLD,
    )
    retained_no_sequence_indices = [no_sequence_features.index(feature) for feature in retained_no_sequence]
    X_train_no_sequence_corr = np.ascontiguousarray(X_train_no_sequence_full[:, retained_no_sequence_indices])
    X_test_no_sequence_corr = np.ascontiguousarray(X_test_no_sequence_full[:, retained_no_sequence_indices])
    ranking_no_sequence = anova_rank(X_train_no_sequence_corr, y_train, retained_no_sequence)
    top10_no_sequence_features = ranking_no_sequence.head(TOP_K)["Feature"].tolist()
    top10_no_sequence_indices = [retained_no_sequence.index(feature) for feature in top10_no_sequence_features]
    X_train_no_sequence_top10 = np.ascontiguousarray(
        X_train_no_sequence_corr[:, top10_no_sequence_indices]
    )
    X_test_no_sequence_top10 = np.ascontiguousarray(
        X_test_no_sequence_corr[:, top10_no_sequence_indices]
    )

    arrays = {
        "X_train_full": X_train_full,
        "X_test_full": X_test_full,
        "X_train_corr": X_train_corr,
        "X_test_corr": X_test_corr,
        "X_train_top10": X_train_top10,
        "X_test_top10": X_test_top10,
        "X_train_full_no_seq_offset": X_train_no_sequence_full,
        "X_test_full_no_seq_offset": X_test_no_sequence_full,
        "X_train_corr_no_seq_offset": X_train_no_sequence_corr,
        "X_test_corr_no_seq_offset": X_test_no_sequence_corr,
        "X_train_top10_no_seq_offset": X_train_no_sequence_top10,
        "X_test_top10_no_seq_offset": X_test_no_sequence_top10,
        "y_train": y_train,
        "y_test": y_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    for name, array in arrays.items():
        save_array(name, array)

    correlation.to_csv(OUT / "training_correlation_matrix.csv")
    high_pairs.to_csv(OUT / "training_high_correlation_pairs.csv", index=False)
    ranking.to_csv(OUT / "training_anova_after_correlation.csv", index=False)
    correlation_no_sequence.to_csv(OUT / "training_correlation_matrix_no_seq_offset.csv")
    pairs_no_sequence.to_csv(OUT / "training_high_correlation_pairs_no_seq_offset.csv", index=False)
    ranking_no_sequence.to_csv(OUT / "training_anova_no_seq_offset.csv", index=False)

    metadata = {
        "data_file": "data/Combined.csv",
        "source": CONFIG["dataset"]["source"],
        "dataset_doi": CONFIG["dataset"]["doi"],
        "dataset_created": CONFIG["dataset"]["created"],
        "dataset_version_last_updated": CONFIG["dataset"]["version_last_updated"],
        "data_sha256": current_hash,
        "raw_records": raw_records,
        "duplicate_rows_removed": int(len(duplicate_positions)),
        "analysis_records": int(len(data)),
        "attack_type_counts_after_deduplication": attack_type_counts,
        "original_columns": len(header),
        "candidate_predictor_count_before_variance": len(predictor_names),
        "candidate_predictors_before_variance": predictor_names,
        "zero_variance_features_training_only": zero_variance_features,
        "usable_predictor_count": len(usable_features),
        "usable_predictors": usable_features,
        "train_records": len(train_indices),
        "test_records": len(test_indices),
        "train_class_counts": {
            "benign": int((y_train == 0).sum()),
            "malicious": int((y_train == 1).sum()),
        },
        "test_class_counts": {
            "benign": int((y_test == 0).sum()),
            "malicious": int((y_test == 1).sum()),
        },
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "numeric_imputation": "median fitted on training only",
        "categorical_imputation": "most frequent fitted on training only",
        "categorical_encoding": "OrdinalEncoder fitted on training only; unknown_value=-1",
        "correlation_threshold_abs": CORR_THRESHOLD,
        "correlation_reduced_count": len(retained),
        "correlation_reduced_features": retained,
        "top10_features": top10_features,
        "no_seq_offset_full_count": len(no_sequence_features),
        "no_seq_offset_correlation_count": len(retained_no_sequence),
        "top10_no_seq_offset_features": top10_no_sequence_features,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
