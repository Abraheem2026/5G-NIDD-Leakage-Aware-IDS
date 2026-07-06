# Reproducibility record

## Verified dataset identity

The code was checked against the original `Combined.csv` file with:

- 1,215,890 raw records;
- 52 columns;
- file size 275,265,610 bytes; and
- SHA-256 `fa36f80859585f474504ca69eb951a079b35145537976c86b00aae3aab46ee59`.

The exact-duplicate audit removed 21 benign records, leaving 1,215,869 observations. The fixed stratified split produced 851,108 development records and 364,761 holdout records.

## Reproduced feature pipeline

The verified execution produced:

- 48 candidate predictors after excluding the exported index, attack labels/tools, and target;
- 46 usable predictors after removing `sVid` and `dVid` as zero-variance features;
- 34 predictors after deterministic correlation reduction; and
- the following development-selected Top-10 representation:

```text
Seq, Offset, sTtl, AckDat, TcpRtt, sMeanPktSz, sHops, Dur, dTtl, SrcBytes
```

## Reproduced scientific results

The stored files in `reference_results/` contain the independently verified values for:

- key holdout conditions;
- fold-specific five-fold cross-validation;
- feature-selection stability;
- ANOVA ranking; and
- exact McNemar tests with Holm correction.

After executing stages 1–4, run:

```bash
python verify_reproduction.py
```

The check uses strict numeric tolerances aligned with the precision reported in the manuscript. Timing is excluded from strict verification because it depends on hardware and runtime conditions.

## Interpretation boundary

The reproduced near-perfect results apply to random record-level evaluation within the same 5G-NIDD collection environment. Removing `Seq` and `Offset` substantially reduces performance. The repository therefore reproduces the manuscript's central conclusion that record-level results must be interpreted in the context of acquisition-order metadata and the partitioning design.
