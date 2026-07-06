# Generated outputs

The pipeline writes generated files to two ignored directories:

- `work/`: intermediate arrays and preprocessing artifacts;
- `results/`: final model results, statistical tests, figures, and tables.

## Stage 1 — preprocessing

Important files under `work/preprocessing/` include:

- `preprocessing_metadata.json`: dataset identity, split sizes, feature counts, and selected features;
- `duplicate_rows_removed.csv`: zero-based positions of exact duplicates removed;
- `duplicate_groups_exact.csv`: exact duplicate groups identified during the audit;
- `duplicate_class_summary.csv`: labels and attack types of the removed duplicate records;
- `class_distribution_after_deduplication.csv`: Table 2 source data;
- `training_variance_filter.csv`: training-only variance audit;
- `training_high_correlation_pairs.csv`: pairs above the absolute 0.90 threshold;
- `training_anova_after_correlation.csv`: development-partition ANOVA ranking;
- `training_anova_no_seq_offset.csv`: ANOVA ranking after removing `Seq` and `Offset`;
- `X_train_*.npy` and `X_test_*.npy`: six feature representations;
- `y_train.npy` and `y_test.npy`: binary targets; and
- `train_indices.npy` and `test_indices.npy`: fixed record-level split indices.

## Stage 2 — holdout evaluation

`results/holdout/` contains:

- `holdout_results.csv`: all metrics for the three models under six feature conditions;
- prediction arrays for the standard and no-Seq/Offset Top-10 conditions; and
- continuous score arrays used for ROC-AUC and PR-AUC.

## Stage 3 — cross-validation

`results/cv/` contains:

- `cv_fold_results.csv`: fold-level model metrics;
- `cv_summary.csv`: mean and sample standard deviation across five folds;
- `cv_selected_features.csv`: selected feature order in every fold;
- `cv_feature_stability.csv`: selection frequency across folds; and
- `cv_preprocessing_times.csv`: preprocessing and feature-selection durations.

## Stage 4 — paired tests

`results/statistics/statistical_tests_mcnemar.csv` contains the discordant counts, exact McNemar p-values, and Holm-adjusted p-values for the nine pre-specified paired comparisons.

## Stage 5 — controlled timing

`results/timing/` contains raw and summarized timing values from three repetitions with a two-thread limit. These values are implementation- and hardware-specific.

## Stage 6 — figures

`results/figures/` contains Figures 1–16 in PNG and PDF formats. Figures 15 and 16 are omitted when the timing stage has not been run.

## Stage 7 — tables

`results/tables/` contains CSV versions of manuscript Tables 2–10. Table 10 is omitted when the timing stage has not been run.
