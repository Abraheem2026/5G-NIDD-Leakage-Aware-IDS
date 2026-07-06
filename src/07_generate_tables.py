from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "work" / "preprocessing"
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["Random Forest", "XGBoost", "RFF-SVM"]


def percentage(series: pd.Series, decimals: int = 4) -> pd.Series:
    return (series.astype(float) * 100).round(decimals)


def main() -> None:
    holdout = pd.read_csv(RESULTS / "holdout" / "holdout_results.csv")
    cv = pd.read_csv(RESULTS / "cv" / "cv_summary.csv")
    tests = pd.read_csv(RESULTS / "statistics" / "statistical_tests_mcnemar.csv")
    ranking = pd.read_csv(PREP / "training_anova_after_correlation.csv")
    distribution = pd.read_csv(PREP / "class_distribution_after_deduplication.csv")

    # Table 2: class/attack distribution after exact-duplicate removal.
    display_names = {
        "Benign": "Benign",
        "UDPFlood": "UDP Flood",
        "HTTPFlood": "HTTP Flood",
        "SlowrateDoS": "Slow-rate DoS",
        "TCPConnectScan": "TCP Connect Scan",
        "SYNScan": "SYN Scan",
        "UDPScan": "UDP Scan",
        "SYNFlood": "SYN Flood",
        "ICMPFlood": "ICMP Flood",
    }
    table2 = distribution[["Attack Type", "records"]].copy()
    table2["Class / attack category"] = table2["Attack Type"].map(display_names).fillna(table2["Attack Type"])
    table2 = table2[["Class / attack category", "records"]].rename(columns={"records": "Records"})
    table2 = pd.concat(
        [table2, pd.DataFrame([{"Class / attack category": "Total", "Records": int(table2["Records"].sum())}])],
        ignore_index=True,
    )
    table2.to_csv(OUT / "table_02_distribution_after_duplicate_removal.csv", index=False)

    # Table 3: development-selected Top-10 features.
    table3 = ranking.head(10).copy()
    table3.insert(0, "Rank", range(1, 11))
    table3 = table3.rename(columns={"Feature": "Feature", "F_Score": "F-score", "p_value": "p-value"})
    table3.to_csv(OUT / "table_03_development_selected_top10.csv", index=False)

    # Table 4: standard holdout feature-set comparison.
    standard = holdout[~holdout["feature_key"].str.contains("no_seq")].copy()
    condition_order = ["Full usable", "Correlation-reduced", "Top-10"]
    standard["feature_condition"] = pd.Categorical(
        standard["feature_condition"], categories=condition_order, ordered=True
    )
    standard["model"] = pd.Categorical(standard["model"], categories=MODEL_ORDER, ordered=True)
    standard = standard.sort_values(["feature_condition", "model"])
    table4 = standard[["feature_condition", "model", "n_features", "accuracy", "f1", "mcc"]].copy()
    table4["Accuracy (%)"] = percentage(table4["accuracy"])
    table4["F1 (%)"] = percentage(table4["f1"])
    table4["MCC"] = table4["mcc"].round(6)
    table4 = table4[["feature_condition", "model", "n_features", "Accuracy (%)", "F1 (%)", "MCC"]]
    table4.columns = ["Feature set", "Model", "No. of predictors", "Accuracy (%)", "F1 (%)", "MCC"]
    table4.to_csv(OUT / "table_04_holdout_feature_set_comparison.csv", index=False)

    # Tables 5 and 6: detailed Top-10 metrics and confusion counts.
    top10 = holdout[holdout["feature_key"] == "top10"].copy()
    top10["model"] = pd.Categorical(top10["model"], categories=MODEL_ORDER, ordered=True)
    top10 = top10.sort_values("model")
    table5 = top10[["model", "accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "mcc"]].copy()
    for source, target in [
        ("accuracy", "Accuracy (%)"),
        ("precision", "Precision (%)"),
        ("recall", "Recall (%)"),
        ("f1", "F1 (%)"),
        ("specificity", "Specificity (%)"),
    ]:
        table5[target] = percentage(table5[source])
    table5["ROC-AUC"] = table5["roc_auc"].round(6)
    table5["PR-AUC"] = table5["pr_auc"].round(6)
    table5["MCC"] = table5["mcc"].round(6)
    table5 = table5[["model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "Specificity (%)", "ROC-AUC", "PR-AUC", "MCC"]]
    table5.columns = ["Model", *table5.columns[1:]]
    table5.to_csv(OUT / "table_05_top10_holdout_performance.csv", index=False)

    table6 = top10[["model", "tn", "fp", "fn", "tp"]].copy()
    table6.columns = ["Model", "TN", "FP", "FN", "TP"]
    table6.to_csv(OUT / "table_06_top10_confusion_counts.csv", index=False)

    # Table 7: holdout performance after removing Seq and Offset.
    no_sequence = holdout[holdout["feature_key"].str.contains("no_seq")].copy()
    no_sequence_condition_order = [
        "Full usable without Seq/Offset",
        "Correlation-reduced without Seq/Offset",
        "Top-10 without Seq/Offset",
    ]
    no_sequence["feature_condition"] = pd.Categorical(
        no_sequence["feature_condition"], categories=no_sequence_condition_order, ordered=True
    )
    no_sequence["model"] = pd.Categorical(no_sequence["model"], categories=MODEL_ORDER, ordered=True)
    no_sequence = no_sequence.sort_values(["feature_condition", "model"])
    table7 = no_sequence[["feature_condition", "model", "n_features", "accuracy", "recall", "f1", "mcc"]].copy()
    table7["Accuracy (%)"] = percentage(table7["accuracy"])
    table7["Recall (%)"] = percentage(table7["recall"])
    table7["F1-score (%)"] = percentage(table7["f1"])
    table7["MCC"] = table7["mcc"].round(4)
    table7 = table7[["feature_condition", "model", "n_features", "Accuracy (%)", "Recall (%)", "F1-score (%)", "MCC"]]
    table7.columns = ["Feature representation", "Model", "No. of predictors", "Accuracy (%)", "Recall (%)", "F1-score (%)", "MCC"]
    table7.to_csv(OUT / "table_07_holdout_without_seq_offset.csv", index=False)

    # Table 8: fold-specific five-fold cross-validation summary.
    cv["condition"] = pd.Categorical(
        cv["condition"], categories=["Top-10", "Top-10 without Seq/Offset"], ordered=True
    )
    cv["model"] = pd.Categorical(cv["model"], categories=MODEL_ORDER, ordered=True)
    cv = cv.sort_values(["condition", "model"])
    table8 = cv[[
        "condition",
        "model",
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
    ]].copy()
    for metric in ["accuracy", "precision", "recall", "f1"]:
        table8[metric.title()] = table8.apply(
            lambda row: f"{row[f'{metric}_mean'] * 100:.4f}% ± {row[f'{metric}_std'] * 100:.4f}%",
            axis=1,
        )
    table8["MCC"] = table8.apply(
        lambda row: f"{row['mcc_mean']:.4f} ± {row['mcc_std']:.4f}", axis=1
    )
    table8 = table8[["condition", "model", "Accuracy", "Precision", "Recall", "F1", "MCC"]]
    table8.columns = ["Condition", "Model", "Accuracy", "Precision", "Recall", "F1", "MCC"]
    table8.to_csv(OUT / "table_08_five_fold_cv.csv", index=False)

    # Table 9: paired statistical tests.
    table9 = tests[[
        "comparison",
        "A_correct_B_wrong",
        "A_wrong_B_correct",
        "exact_p_value",
        "holm_adjusted_p_value",
    ]].copy()
    table9.columns = [
        "Comparison",
        "A correct / B wrong",
        "A wrong / B correct",
        "Exact p-value",
        "Holm-adjusted p-value",
    ]
    table9.to_csv(OUT / "table_09_mcnemar_holm.csv", index=False)

    # Table 10: controlled timing, when stage 5 was run.
    timing_file = RESULTS / "timing" / "fair_timing_results.csv"
    if timing_file.exists():
        timing = pd.read_csv(timing_file)
        timing["model"] = pd.Categorical(timing["model"], categories=MODEL_ORDER, ordered=True)
        summary = timing.groupby("model", observed=False)[
            ["training_time_s", "prediction_time_s", "latency_ms_per_1000"]
        ].agg(["mean", "std"])
        rows = []
        for model in MODEL_ORDER:
            rows.append(
                {
                    "Model": model,
                    "Training time (s)": f"{summary.loc[model, ('training_time_s', 'mean')]:.3f} ± {summary.loc[model, ('training_time_s', 'std')]:.3f}",
                    "Prediction time (s)": f"{summary.loc[model, ('prediction_time_s', 'mean')]:.3f} ± {summary.loc[model, ('prediction_time_s', 'std')]:.3f}",
                    "Prediction latency per 1,000 records (ms)": f"{summary.loc[model, ('latency_ms_per_1000', 'mean')]:.3f} ± {summary.loc[model, ('latency_ms_per_1000', 'std')]:.3f}",
                }
            )
        pd.DataFrame(rows).to_csv(OUT / "table_10_controlled_timing.csv", index=False)

    print(f"Tables written to: {OUT}")


if __name__ == "__main__":
    main()
