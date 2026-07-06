from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "work" / "preprocessing"
HOLDOUT = ROOT / "results" / "holdout"
OUT = ROOT / "results" / "statistics"
OUT.mkdir(parents=True, exist_ok=True)

y_true = np.load(PREP / "y_test.npy")


def exact_mcnemar(name: str, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict[str, object]:
    correct_a = prediction_a == y_true
    correct_b = prediction_b == y_true
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = (
        float(
            binomtest(
                min(a_correct_b_wrong, a_wrong_b_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "comparison": name,
        "A_correct_B_wrong": a_correct_b_wrong,
        "A_wrong_B_correct": a_wrong_b_correct,
        "discordant_total": discordant,
        "exact_p_value": p_value,
    }


def holm_adjust(rows: list[dict[str, object]]) -> None:
    p_values = np.array([float(row["exact_p_value"]) for row in rows])
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = float(value)


def main() -> None:
    rows: list[dict[str, object]] = []
    for key in ["top10", "top10_no_seq_offset"]:
        predictions = {
            "RF": np.load(HOLDOUT / f"pred_rf_{key}.npy"),
            "XGBoost": np.load(HOLDOUT / f"pred_xgb_{key}.npy"),
            "RFF-SVM": np.load(HOLDOUT / f"pred_rff_{key}.npy"),
        }
        rows.extend(
            [
                exact_mcnemar(f"{key}: RF vs XGBoost", predictions["RF"], predictions["XGBoost"]),
                exact_mcnemar(f"{key}: RF vs RFF-SVM", predictions["RF"], predictions["RFF-SVM"]),
                exact_mcnemar(
                    f"{key}: XGBoost vs RFF-SVM",
                    predictions["XGBoost"],
                    predictions["RFF-SVM"],
                ),
            ]
        )

    for model, slug in [("RF", "rf"), ("XGBoost", "xgb"), ("RFF-SVM", "rff")]:
        standard = np.load(HOLDOUT / f"pred_{slug}_top10.npy")
        ablation = np.load(HOLDOUT / f"pred_{slug}_top10_no_seq_offset.npy")
        rows.append(exact_mcnemar(f"{model}: Top-10 vs Top-10 without Seq/Offset", standard, ablation))

    holm_adjust(rows)
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "statistical_tests_mcnemar.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
