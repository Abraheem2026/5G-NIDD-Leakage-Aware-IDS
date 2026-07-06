# 5G-NIDD Leakage-Aware Intrusion Detection

Reproducibility code for the manuscript:

**Feature-Selected Machine Learning for 5G Intrusion Detection: Leakage-Aware Evaluation and Sensitivity to Sequence Metadata in 5G-NIDD**

Authors: **Abraheem Sulayman Alsaedi Abraheem**, **Ali Saed Riheel Arhoumah**, **Hamza Saleh Ali Fokla**, and **Majeda M. Abdosalam Basheer**.

## Overview

This repository implements the complete experimental pipeline used in the manuscript. It includes:

- exact-duplicate auditing;
- split-before-processing evaluation;
- training-only imputation and categorical encoding;
- zero-variance filtering;
- deterministic Pearson-correlation reduction;
- ANOVA F-test feature ranking;
- Random Forest, XGBoost, and scalable RFF-SVM models;
- an ablation study removing `Seq` and `Offset`;
- fold-specific five-fold cross-validation;
- exact paired McNemar tests with Holm correction;
- controlled two-thread timing; and
- generation of the manuscript's data-driven tables and figures.

The code was independently rerun on the verified original dataset file. The reported holdout, cross-validation, feature-selection, and statistical-test results were reproduced after the rounding used in the manuscript.

## Dataset

The dataset is not redistributed in this repository.

- **Dataset:** 5G-NIDD: A Comprehensive Network Intrusion Detection Dataset Generated over 5G Wireless Network
- **Repository:** IEEE DataPort
- **DOI:** `10.21227/xtep-hv36`
- **File:** `Combined.csv`
- **Raw records:** `1,215,890`
- **Columns:** `52`
- **SHA-256:** `fa36f80859585f474504ca69eb951a079b35145537976c86b00aae3aab46ee59`
- **Dataset created:** 2 December 2022
- **Version used:** last updated 1 January 2026

Download the dataset from IEEE DataPort and place it at:

```text
data/Combined.csv
```

The pipeline verifies the row count, column count, and SHA-256 hash before running.

## Repository structure

```text
.
├── .github/workflows/syntax-check.yml
├── data/
│   └── README.md
├── docs/
│   ├── OUTPUTS.md
│   └── REPRODUCIBILITY.md
├── reference_results/
│   ├── expected_anova_ranking.csv
│   ├── expected_cv_feature_stability.csv
│   ├── expected_cv_summary.csv
│   ├── expected_key_results.csv
│   └── expected_mcnemar_holm.csv
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_holdout_models.py
│   ├── 03_cross_validation.py
│   ├── 04_statistical_tests.py
│   ├── 05_timing_benchmark.py
│   ├── 06_generate_figures.py
│   └── 07_generate_tables.py
├── check_dataset.py
├── verify_reproduction.py
├── config.json
├── requirements.txt
└── run_all.py
```

Generated intermediate files are written to `work/`; final outputs are written to `results/`. Both directories are ignored by Git.

## Software environment

The manuscript experiments used:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- scikit-learn 1.8.0
- XGBoost 3.1.3
- SciPy 1.17.0
- Matplotlib 3.10.8

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify the dataset

```bash
python check_dataset.py
```

Expected final line:

```text
PASS: dataset matches the version used in the paper.
```

## Run the complete pipeline

```bash
python run_all.py
```

The dataset check is performed automatically. To remove previous generated files before a fresh run:

```bash
python run_all.py --clean
```

Run a selected range of stages:

```bash
python run_all.py --from-stage 2 --to-stage 4
```

Skip the hardware-dependent timing benchmark while running the remaining stages:

```bash
python run_all.py --skip-timing
```

### Pipeline stages

1. Dataset audit, preprocessing, and feature selection
2. Holdout model evaluation
3. Fold-specific five-fold cross-validation
4. Exact McNemar tests and Holm correction
5. Controlled timing benchmark
6. Generate figures
7. Generate manuscript tables

## Verify reproduced results

After running stages 1–4, compare the generated results with the verified reference outputs:

```bash
python verify_reproduction.py
```

The verification script checks:

- dataset identity and audited record counts;
- the development-selected Top-10 features;
- key holdout results before and after removing `Seq` and `Offset`;
- five-fold cross-validation summary values;
- feature-selection stability; and
- exact McNemar/Holm results.

Timing values are not used as strict reproduction targets because they depend on hardware, operating system, and library/runtime conditions.

## Key verified results

| Condition | Model | Accuracy | F1 | MCC |
|---|---|---:|---:|---:|
| Top-10 | Random Forest | 99.9745% | 99.9790% | 0.999466 |
| Top-10 | XGBoost | 99.9748% | 99.9792% | 0.999471 |
| Top-10 | RFF-SVM | 99.4939% | 99.5832% | 0.989392 |
| Top-10 without Seq/Offset | Random Forest | 71.1644% | 68.8686% | 0.5499 |
| Top-10 without Seq/Offset | XGBoost | 71.1787% | 68.8771% | 0.5504 |
| Top-10 without Seq/Offset | RFF-SVM | 73.0870% | 77.2431% | 0.4447 |

## Computational notes

The full workflow processes more than 1.2 million records and creates memory-mapped/intermediate NumPy arrays. Runtime and memory requirements vary by system. The timing stage is intentionally restricted to two threads and uses three repetitions with seeds 42, 43, and 44, matching the manuscript protocol.

The RFF-SVM implementation is a scalable approximation to an RBF-kernel SVM. It uses standardized inputs, 300 random Fourier features, and an averaged SGD classifier trained incrementally for five epochs.

## Data and code availability

The original dataset must be obtained separately from IEEE DataPort under its applicable access conditions. This repository does not redistribute `Combined.csv`.

## Citation

Until the associated manuscript receives its final bibliographic record, cite this software using the metadata in `CITATION.cff`. Also cite the original 5G-NIDD dataset using DOI `10.21227/xtep-hv36`.

## License

The code in this repository is released under the MIT License. The dataset is governed separately by the terms of its original IEEE DataPort record.
