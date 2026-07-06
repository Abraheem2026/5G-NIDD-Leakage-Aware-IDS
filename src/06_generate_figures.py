from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "work" / "preprocessing"
RESULTS = ROOT / "results"
HOLDOUT = RESULTS / "holdout"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def workflow_figure() -> None:
    steps = [
        "Official Combined.csv\nfile and identity check",
        "Exact-duplicate audit\nand removal",
        "70:30 stratified split\nbefore preprocessing",
        "Training-only imputation, encoding,\nvariance and correlation filtering",
        "ANOVA Top-10 selection\ninside development data/folds",
        "RF, XGBoost and RFF-SVM\nunder six feature conditions",
        "Holdout, five-fold CV, ablation,\nMcNemar/Holm and timing",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ys = np.linspace(0.91, 0.09, len(steps))
    for index, (label, y) in enumerate(zip(steps, ys)):
        box = FancyBboxPatch(
            (0.14, y - 0.045),
            0.72,
            0.09,
            boxstyle="round,pad=0.012",
            linewidth=1.2,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(0.5, y, label, ha="center", va="center")
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, ys[index + 1] + 0.052),
                xytext=(0.5, y - 0.052),
                arrowprops={"arrowstyle": "->", "linewidth": 1.2},
            )
    ax.set_title("Unified leakage-aware experimental workflow", pad=14)
    save(fig, "figure_01_workflow.png")


def audit_figure(metadata: dict[str, object]) -> None:
    labels = [
        "Raw records",
        "After duplicate removal",
        "Development partition",
        "Holdout partition",
    ]
    records = [
        int(metadata["raw_records"]),
        int(metadata["analysis_records"]),
        int(metadata["train_records"]),
        int(metadata["test_records"]),
    ]
    feature_labels = ["Candidate", "Usable", "Correlation-reduced", "Top-10"]
    feature_counts = [
        int(metadata["candidate_predictor_count_before_variance"]),
        int(metadata["usable_predictor_count"]),
        int(metadata["correlation_reduced_count"]),
        len(metadata["top10_features"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    axes[0].bar(labels, records)
    axes[0].set_ylabel("Records")
    axes[0].set_title("Record-level audit")
    axes[0].tick_params(axis="x", rotation=25)
    for index, value in enumerate(records):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(feature_labels, feature_counts)
    axes[1].set_ylabel("Predictors")
    axes[1].set_title("Feature-space reduction")
    axes[1].tick_params(axis="x", rotation=25)
    for index, value in enumerate(feature_counts):
        axes[1].text(index, value, str(value), ha="center", va="bottom")
    save(fig, "figure_02_record_feature_audit.png")


def confusion_figure(y_true: np.ndarray, suffix: str, figure_number: int, title: str) -> None:
    models = [("Random Forest", "rf"), ("XGBoost", "xgb"), ("RFF-SVM", "rff")]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, (model_name, slug) in zip(axes, models):
        prediction = np.load(HOLDOUT / f"pred_{slug}_{suffix}.npy")
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malicious"]).plot(
            ax=ax,
            colorbar=False,
            values_format="d",
        )
        ax.set_title(model_name)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"figure_{figure_number:02d}_confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"figure_{figure_number:02d}_confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def timing_figures() -> None:
    timing_file = RESULTS / "timing" / "fair_timing_results.csv"
    if not timing_file.exists():
        warnings.warn("Timing results are missing; Figures 15 and 16 were not generated.")
        return

    timing = pd.read_csv(timing_file)
    summary = timing.groupby("model")[["training_time_s", "prediction_time_s"]].agg(["mean", "std"])
    order = ["Random Forest", "XGBoost", "RFF-SVM"]

    for figure_number, metric, ylabel, title in [
        (15, "training_time_s", "Training time (s)", "Controlled two-thread training time"),
        (16, "prediction_time_s", "Prediction time (s)", "Controlled two-thread prediction time"),
    ]:
        means = summary[(metric, "mean")].reindex(order)
        stds = summary[(metric, "std")].reindex(order)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(order, means, yerr=stds, capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(fig, f"figure_{figure_number:02d}_{metric}.png")


def main() -> None:
    metadata = json.loads((PREP / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    holdout = pd.read_csv(HOLDOUT / "holdout_results.csv")
    cv = pd.read_csv(RESULTS / "cv" / "cv_summary.csv")
    anova = pd.read_csv(PREP / "training_anova_after_correlation.csv").head(10)
    correlation_pairs = pd.read_csv(PREP / "training_high_correlation_pairs.csv")
    stability = pd.read_csv(RESULTS / "cv" / "cv_feature_stability.csv")
    y_test = np.load(PREP / "y_test.npy")

    workflow_figure()
    audit_figure(metadata)

    pairs = correlation_pairs.copy()
    pairs["pair"] = pairs["Feature_A"] + " / " + pairs["Feature_B"]
    pairs = pairs.sort_values("Pearson_r")
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.barh(pairs["pair"], pairs["Pearson_r"].abs())
    ax.axvline(0.90, linestyle="--", linewidth=1)
    ax.set_xlim(0.88, 1.005)
    ax.set_xlabel("|Pearson r|")
    ax.set_title("Training-only feature pairs above the 0.90 threshold")
    save(fig, "figure_03_high_correlation_pairs.png")

    ranked = anova.sort_values("F_Score")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(ranked["Feature"], ranked["F_Score"])
    ax.set_xlabel("ANOVA F-score")
    ax.set_title("Top-10 features selected from the development partition")
    save(fig, "figure_04_anova_top10.png")

    standard = holdout[~holdout["feature_key"].str.contains("no_seq")].copy()
    for figure_number, metric, title in [
        (5, "accuracy", "Holdout accuracy across feature representations"),
        (6, "f1", "Holdout F1-score across feature representations"),
    ]:
        pivot = standard.pivot(index="feature_condition", columns="model", values=metric).loc[
            ["Full usable", "Correlation-reduced", "Top-10"]
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(max(0.98, pivot.min().min() - 0.002), 1.0002)
        ax.set_ylabel(metric.upper() if metric == "f1" else metric.title())
        ax.set_xlabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=0)
        save(fig, f"figure_{figure_number:02d}_holdout_{metric}.png")

    confusion_figure(y_test, "top10", 7, "Top-10 holdout confusion matrices")

    ablation = holdout[holdout["feature_key"].isin(["top10", "top10_no_seq_offset"])].copy()
    ablation["Condition"] = ablation["feature_key"].map(
        {"top10": "Top-10", "top10_no_seq_offset": "Top-10 without Seq/Offset"}
    )
    for figure_number, metric, title in [
        (8, "accuracy", "Effect of removing Seq and Offset on accuracy"),
        (9, "recall", "Effect of removing Seq and Offset on recall"),
        (10, "mcc", "Effect of removing Seq and Offset on MCC"),
    ]:
        pivot = ablation.pivot(index="Condition", columns="model", values=metric).loc[
            ["Top-10", "Top-10 without Seq/Offset"]
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(metric.upper() if metric == "mcc" else metric.title())
        ax.set_xlabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=0)
        save(fig, f"figure_{figure_number:02d}_ablation_{metric}.png")

    confusion_figure(
        y_test,
        "top10_no_seq_offset",
        11,
        "Top-10 holdout confusion matrices without Seq/Offset",
    )

    for figure_number, metric, title in [
        (12, "accuracy", "Fold-specific five-fold CV accuracy"),
        (13, "f1", "Fold-specific five-fold CV F1-score"),
    ]:
        conditions = ["Top-10", "Top-10 without Seq/Offset"]
        models = ["Random Forest", "XGBoost", "RFF-SVM"]
        x = np.arange(len(conditions))
        width = 0.22
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for index, model in enumerate(models):
            subset = cv[cv["model"] == model].set_index("condition").loc[conditions]
            ax.bar(
                x + (index - 1) * width,
                subset[f"{metric}_mean"],
                width,
                yerr=subset[f"{metric}_std"],
                capsize=3,
                label=model,
            )
        ax.set_xticks(x, conditions)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(metric.upper() if metric == "f1" else metric.title())
        ax.set_title(title)
        ax.legend(fontsize=8)
        save(fig, f"figure_{figure_number:02d}_cv_{metric}.png")

    stable = stability[stability["condition"] == "Top-10 without Seq/Offset"].sort_values(
        ["selection_frequency", "feature"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(stable["feature"], stable["selection_frequency"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Selection frequency across five folds")
    ax.set_title("Feature-selection stability after removing Seq and Offset")
    save(fig, "figure_14_feature_stability.png")

    timing_figures()
    print(f"Figures written to: {OUT}")


if __name__ == "__main__":
    main()
