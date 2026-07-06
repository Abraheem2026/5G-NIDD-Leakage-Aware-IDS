from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PREP = ROOT / "work" / "preprocessing"
RESULTS = ROOT / "results"
REFERENCE = ROOT / "reference_results"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required generated file is missing: {path.relative_to(ROOT)}. "
            "Run the relevant pipeline stages first."
        )
    return path


def close(actual: float, expected: float, atol: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True))


def main() -> None:
    failures: list[str] = []
    passes: list[str] = []

    metadata = json.loads(require(PREP / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "data_sha256": "fa36f80859585f474504ca69eb951a079b35145537976c86b00aae3aab46ee59",
        "raw_records": 1_215_890,
        "duplicate_rows_removed": 21,
        "analysis_records": 1_215_869,
        "train_records": 851_108,
        "test_records": 364_761,
        "usable_predictor_count": 46,
        "correlation_reduced_count": 34,
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual == expected:
            passes.append(f"metadata.{key}")
        else:
            failures.append(f"metadata.{key}: expected {expected!r}, got {actual!r}")

    expected_ranking = pd.read_csv(REFERENCE / "expected_anova_ranking.csv")
    actual_ranking = pd.read_csv(require(PREP / "training_anova_after_correlation.csv"))
    expected_top10 = expected_ranking.head(10)["Feature"].tolist()
    actual_top10 = actual_ranking.head(10)["Feature"].tolist()
    if actual_top10 == expected_top10:
        passes.append("development-selected Top-10 feature order")
    else:
        failures.append(f"Top-10 mismatch: expected {expected_top10}, got {actual_top10}")

    holdout = pd.read_csv(require(RESULTS / "holdout" / "holdout_results.csv"))
    expected_key = pd.read_csv(REFERENCE / "expected_key_results.csv")
    for _, row in expected_key.iterrows():
        selected = holdout[
            (holdout["feature_key"] == row["condition"])
            & (holdout["model"] == row["model"])
        ]
        label = f"holdout {row['condition']} / {row['model']}"
        if len(selected) != 1:
            failures.append(f"{label}: expected one generated row, found {len(selected)}")
            continue
        actual = selected.iloc[0]
        checks = [
            close(actual["accuracy"] * 100, row["paper_accuracy_pct"], 5e-5),
            close(actual["f1"] * 100, row["paper_f1_pct"], 5e-5),
            close(actual["mcc"], row["paper_mcc"], 5e-5),
        ]
        if all(checks):
            passes.append(label)
        else:
            failures.append(
                f"{label}: generated accuracy/F1/MCC = "
                f"{actual['accuracy'] * 100:.8f}, {actual['f1'] * 100:.8f}, {actual['mcc']:.8f}"
            )

    expected_cv = pd.read_csv(REFERENCE / "expected_cv_summary.csv")
    actual_cv = pd.read_csv(require(RESULTS / "cv" / "cv_summary.csv"))
    cv_metrics = [
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
    ]
    for _, row in expected_cv.iterrows():
        selected = actual_cv[
            (actual_cv["condition"] == row["condition"])
            & (actual_cv["model"] == row["model"])
        ]
        label = f"CV {row['condition']} / {row['model']}"
        if len(selected) != 1:
            failures.append(f"{label}: expected one generated row, found {len(selected)}")
            continue
        actual = selected.iloc[0]
        mismatches = [
            metric
            for metric in cv_metrics
            if not close(float(actual[metric]), float(row[metric]), 5e-7)
        ]
        if not mismatches:
            passes.append(label)
        else:
            failures.append(f"{label}: mismatched metrics {mismatches}")

    expected_stability = pd.read_csv(REFERENCE / "expected_cv_feature_stability.csv").sort_values(
        ["condition", "feature"]
    ).reset_index(drop=True)
    actual_stability = pd.read_csv(require(RESULTS / "cv" / "cv_feature_stability.csv")).sort_values(
        ["condition", "feature"]
    ).reset_index(drop=True)
    stability_columns = ["condition", "feature", "selection_count", "selection_frequency"]
    if actual_stability[stability_columns].equals(expected_stability[stability_columns]):
        passes.append("cross-validation feature-selection stability")
    else:
        failures.append("cross-validation feature-selection stability differs from the verified reference")

    expected_tests = pd.read_csv(REFERENCE / "expected_mcnemar_holm.csv")
    actual_tests = pd.read_csv(require(RESULTS / "statistics" / "statistical_tests_mcnemar.csv"))
    for _, row in expected_tests.iterrows():
        selected = actual_tests[actual_tests["comparison"] == row["comparison"]]
        label = f"McNemar/Holm {row['comparison']}"
        if len(selected) != 1:
            failures.append(f"{label}: expected one generated row, found {len(selected)}")
            continue
        actual = selected.iloc[0]
        count_match = (
            int(actual["A_correct_B_wrong"]) == int(row["A_correct_B_wrong"])
            and int(actual["A_wrong_B_correct"]) == int(row["A_wrong_B_correct"])
            and int(actual["discordant_total"]) == int(row["discordant_total"])
        )
        p_match = close(float(actual["exact_p_value"]), float(row["exact_p_value"]), 1e-12)
        holm_match = close(
            float(actual["holm_adjusted_p_value"]),
            float(row["holm_adjusted_p_value"]),
            1e-12,
        )
        if count_match and p_match and holm_match:
            passes.append(label)
        else:
            failures.append(f"{label}: generated values differ from the verified reference")

    print(f"PASS checks: {len(passes)}")
    for item in passes:
        print(f"  [PASS] {item}")

    if failures:
        print(f"\nFAIL checks: {len(failures)}")
        for item in failures:
            print(f"  [FAIL] {item}")
        raise SystemExit(1)

    print("\nREPRODUCTION VERIFIED: generated scientific results match the stored verified references.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"Verification could not be completed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
