# Publishing the repository and archiving a release

## 1. Create the GitHub repository

Recommended repository name:

```text
5G-NIDD-Leakage-Aware-IDS
```

Create an empty public repository without automatically adding a README, license, or `.gitignore`, because those files are already included here.

## 2. Upload with Git

From the repository directory:

```bash
git init
git add .
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/5G-NIDD-Leakage-Aware-IDS.git
git push -u origin main
```

Verify on GitHub that `data/Combined.csv`, `work/`, and `results/` were not uploaded.

## 3. Create a release

Create a GitHub release with tag:

```text
v1.0.0
```

Suggested release title:

```text
Reproducibility release for the 5G-NIDD leakage-aware evaluation study
```

## 4. Archive through Zenodo

1. Sign in to Zenodo using GitHub.
2. Enable the new repository in Zenodo's GitHub integration.
3. Publish or re-publish the GitHub release.
4. Zenodo will archive the release and assign a software DOI.
5. Add the Zenodo DOI badge and DOI to the README and manuscript code-availability statement.

## 5. Final checks

- Confirm the authors and name parsing in `CITATION.cff`.
- Confirm the MIT license is the intended code license.
- Confirm that the dataset file is absent from Git history.
- Confirm that the GitHub release archive contains no generated `work/` or `results/` directories.
- Run the syntax-check workflow successfully.
