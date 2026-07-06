# Verified reference results

These small files record the independently reproduced scientific outputs from the verified original `Combined.csv` dataset. They allow `verify_reproduction.py` to check a fresh pipeline run without redistributing the dataset or large intermediate arrays.

Included checks cover:

- development-partition ANOVA ranking;
- key holdout results before and after removing `Seq` and `Offset`;
- five-fold cross-validation summaries;
- fold-level feature-selection stability; and
- exact McNemar tests with Holm correction.

Hardware-dependent timing values are intentionally excluded from strict reference verification.
