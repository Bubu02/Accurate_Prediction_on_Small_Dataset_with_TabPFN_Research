# Accurate Prediction on Small Dataset with TabPFN Research

## Overview

This repository documents a research-driven benchmark study on **TabPFN** for **small tabular datasets**, combining:

* theoretical reading and notes on the TabPFN method,
* practical benchmarking on classification and regression datasets,
* aggregated comparison tables across multiple baseline models,
* notebook and HTML-based experiment outputs,
* presentation assets for communicating the findings.

The central aim of the project is to evaluate whether **TabPFN** delivers strong predictive performance on small datasets when compared against commonly used tabular ML baselines such as **CatBoost, LightGBM, XGBoost, Random Forest, Logistic Regression, SVM, MLP, AutoGluon, and AutoTabPFN**.

---

## Why this project matters

Small tabular datasets are common in biomedical, academic, and domain-specific problems where collecting large volumes of labeled data is difficult. TabPFN is designed specifically for this setting, and this repository explores its behavior in two practical directions:

1. **Classification** on multiple small benchmark datasets.
2. **Regression** on multiple small benchmark datasets.

This makes the repository useful for students, researchers, and practitioners who want to understand how TabPFN performs in realistic small-data settings rather than only in theory.

---

## Project goals

The repository appears to pursue four main goals:

1. **Understand TabPFN conceptually** through theory notes and literature review.
2. **Evaluate TabPFN empirically** on several small datasets.
3. **Compare TabPFN against strong classical and AutoML baselines**.
4. **Summarize results in a report-ready form** using CSV, LaTeX, HTML, spreadsheet, and document outputs.

---

## Repository structure

A high-level interpretation of the repository structure is below.

```text
.
├── HTML View/
├── Practical Research/
│   ├── Results/
│   │   ├── Classification/
│   │   └── Regression/
│   └── notebooks / experiments
├── Presentation/
├── Theoretical Research/
├── README.md
└── auxiliary/editor files
```

### Main folders

#### `Theoretical Research/`

Contains theory-oriented study material related to TabPFN. This is useful for understanding the conceptual basis of the project before looking at experiments.

#### `Practical Research/`

This is the core experimental section of the repository. It contains notebooks, result tables, and derived artifacts for classification and regression studies.

#### `Practical Research/Results/Classification/`

Contains dataset-level and aggregated classification benchmarking outputs. This includes:

* per-dataset comparison CSV files,
* summary tables,
* aggregation scripts,
* tuned-result files,
* LaTeX-style output for reporting.

#### `Practical Research/Results/Regression/`

Contains regression experiment outputs, including:

* aggregated regression comparison tables,
* spreadsheet exports,
* PDF and document versions of result reports.

#### `HTML View/`

Stores HTML-exported notebook views for easier browsing without opening Jupyter directly.

#### `Presentation/`

Contains presentation planning material, suggesting the project was also prepared for academic or seminar presentation.

---

## Research workflow represented in this repo

The repository suggests the following workflow:

1. **Study TabPFN theoretically**.
2. **Prepare tabular datasets** for classification and regression.
3. **Run multiple baseline models**.
4. **Evaluate raw and processed variants** of the data.
5. **Aggregate metrics across datasets**.
6. **Compare TabPFN with other methods** using summary tables.
7. **Prepare outputs for reporting** in HTML, CSV, LaTeX, Excel, PDF, and DOCX formats.

This is a strong workflow for an academic benchmarking project because it separates understanding, experimentation, aggregation, and communication.

---

## Models compared in this project

Across the classification and regression outputs, the following models are compared:

* **TabPFN**
* **AutoTabPFN**
* **AutoGluon**
* **CatBoost**
* **LightGBM**
* **XGBoost**
* **RandomForest**
* **LogisticRegression** (classification)
* **LinearRegression / Ridge / SVR / GradientBoosting** (regression)
* **MLP**
* **SVM / SVC**

This is a good comparison set because it includes:

* classical linear baselines,
* tree-based strong baselines,
* neural baselines,
* AutoML baselines,
* TabPFN-family methods.

---

## Evaluation design used in the repository

The result files show that the experiments compare models under two settings:

* **Raw** data
* **Processed** data

This is an important design choice because it evaluates whether preprocessing improves or changes relative model behavior.

### Classification metrics tracked

The classification outputs include:

* ROC AUC
* Accuracy
* F1
* CE (cross-entropy / log-loss style metric)
* ECE (expected calibration error)
* Fit / runtime

### Regression metrics tracked

The regression outputs include:

* RMSE
* R²
* MAE
* win counts across datasets
* fit / runtime

This is a strong metric set because it covers both:

* **predictive quality**, and
* **efficiency / practicality**.

---

## Key result summary

## 1) Classification results

The aggregated classification results indicate that **TabPFN is the strongest overall model family in normalized performance**, with **AutoTabPFN** being similarly strong but much slower.

### Classification summary highlights

#### TabPFN

* Raw normalized ROC AUC: **0.997 ± 0.01**

* Raw normalized Accuracy: **0.980 ± 0.03**

* Raw normalized F1: **0.984 ± 0.02**

* Raw normalized CE: **0.989 ± 0.02**

* Raw time: **0.283 ± 0.13 s**

* Processed normalized ROC AUC: **0.995 ± 0.01**

* Processed normalized Accuracy: **0.982 ± 0.03**

* Processed normalized F1: **0.985 ± 0.01**

* Processed normalized CE: **0.984 ± 0.03**

* Processed time: **2.074 ± 0.27 s**

#### AutoTabPFN

* Raw normalized ROC AUC: **0.993 ± 0.01**

* Raw normalized Accuracy: **0.980 ± 0.03**

* Raw normalized F1: **0.986 ± 0.02**

* Raw time: **291.358 ± 291.82 s**

* Processed normalized ROC AUC: **0.994 ± 0.01**

* Processed normalized Accuracy: **0.979 ± 0.03**

* Processed normalized F1: **0.988 ± 0.01**

* Processed time: **292.512 ± 288.35 s**

#### Strong classical baselines

**CatBoost**

* Raw normalized ROC AUC: **0.975 ± 0.03**
* Processed normalized Accuracy: **0.987 ± 0.02**
* Runtime: about **1.5 s**

**LightGBM**

* Raw normalized ROC AUC: **0.959 ± 0.06**
* Processed normalized Accuracy: **0.967 ± 0.06**
* Runtime: about **0.09 s**

**AutoGluon**

* Raw normalized ROC AUC: **0.971 ± 0.05**
* Processed normalized Accuracy: **0.961 ± 0.03**
* Runtime: about **15–16 s**

### Interpretation

The classification results suggest:

* **TabPFN is the best balanced performer overall** on small classification datasets.
* **AutoTabPFN is also very strong**, but its runtime is much larger.
* **CatBoost remains the strongest classical baseline** and is more practical in some runtime-sensitive scenarios.
* **LightGBM and XGBoost are extremely fast**, but they do not match TabPFN’s normalized aggregate performance.
* **TabPFN appears especially strong on CE/log-loss style behavior**, suggesting better confidence quality than several baselines.

---

## 2) Regression results

The regression aggregated results show an even clearer pattern: **TabPFN is the strongest overall model in normalized regression performance**, while also remaining dramatically faster than AutoTabPFN and AutoGluon.

### Regression summary highlights

#### TabPFN

* Raw normalized RMSE: **0.503 ± 0.26**
* Raw normalized R²: **0.998 ± 0.00**
* Raw normalized MAE: **0.447 ± 0.27**
* Processed normalized RMSE: **0.503 ± 0.27**
* Processed normalized R²: **0.995 ± 0.00**
* Processed normalized MAE: **0.448 ± 0.27**
* Fit time: **0.362 ± 0.06 s (raw)**, **0.331 ± 0.05 s (processed)**

#### AutoTabPFN

* Raw normalized RMSE: **0.508 ± 0.28**
* Raw normalized R²: **0.987 ± 0.02**
* Raw normalized MAE: **0.447 ± 0.28**
* Processed normalized RMSE: **0.502 ± 0.27**
* Processed normalized R²: **0.993 ± 0.01**
* Processed normalized MAE: **0.451 ± 0.27**
* Fit time: **192.868 ± 127.52 s**, **180.366 ± 116.19 s**

#### AutoGluon

* Raw normalized RMSE: **0.525 ± 0.29**
* Raw normalized R²: **0.971 ± 0.03**
* Raw normalized MAE: **0.483 ± 0.30**
* Fit time: **69.867 ± 68.54 s**

#### Tree-based regression baselines

**XGBoost**

* Raw normalized RMSE: **0.548 ± 0.30**
* Raw normalized R²: **0.950 ± 0.05**
* Raw normalized MAE: **0.509 ± 0.30**
* Fit time: **0.108 ± 0.05 s**

**LightGBM**

* Raw normalized RMSE: **0.587 ± 0.24**
* Raw normalized R²: **0.953 ± 0.03**
* Raw normalized MAE: **0.554 ± 0.25**
* Fit time: **0.098 ± 0.03 s**

**RandomForest**

* Raw normalized RMSE: **0.592 ± 0.26**
* Raw normalized R²: **0.953 ± 0.04**
* Raw normalized MAE: **0.555 ± 0.27**
* Fit time: **0.347 ± 0.18 s**

### Interpretation

The regression results indicate:

* **TabPFN is the best overall regression model in this benchmark set**.
* It achieves the strongest normalized error scores while maintaining **very low runtime**.
* **AutoTabPFN is competitive but computationally expensive**.
* **AutoGluon is stronger than many conventional baselines**, but still behind TabPFN.
* Conventional tree and linear baselines remain useful because they are fast and interpretable, but they underperform in aggregate.
* The weakest regression models here are **MLP**, **SVR**, and the linear family under these settings.

---

## 3) Heart disease case study

The repository includes a detailed heart disease classification result file and tuned baseline comparisons.

### Best observed heart-disease result pattern

Several models tie at nearly identical top-end accuracy, but **TabPFN stands out through its confidence quality and speed-quality balance**.

#### TabPFN on heart dataset

* ROC AUC: **1.0000**
* Accuracy: **0.9854**
* F1: **0.9852**
* CE: **0.0162**
* Runtime: **0.214 s**

#### AutoTabPFN on heart dataset

* ROC AUC: **1.0000**
* Accuracy: **0.9854**
* F1: **0.9852**
* CE: **0.0731**
* Runtime: **189.277 s**

#### CatBoost on heart dataset

* ROC AUC: **0.9957**
* Accuracy: **0.9854**
* F1: **0.9852**
* CE: **0.0822**
* Runtime: **1.264 s**

#### LightGBM on heart dataset

* ROC AUC: **0.9906**
* Accuracy: **0.9854**
* F1: **0.9852**
* CE: **0.0831**
* Runtime: **0.073 s**

### Tuned baseline observations on heart dataset

* **CatBoost (tuned)**: ROC AUC **1.0**, logloss **0.0257**, fit time **306.76 s**
* **RandomForest (tuned)**: ROC AUC **1.0**, logloss **0.0664**, fit time **7.63 s**
* **XGBoost (tuned)**: ROC AUC **0.9997**, but accuracy collapses to **0.5024**, suggesting poor threshold behavior or a bad hyperparameter configuration

### Interpretation

This case study strongly supports the main conclusion of the repository:

* **TabPFN can match or exceed the strongest baselines on small classification tasks**.
* It does so with **excellent predictive quality and very strong CE/log-loss behavior**.
* On this dataset, **AutoTabPFN is not worth the extra runtime cost** compared with vanilla TabPFN.

---

## What this repository does well

### Strengths

1. **Good research direction**

   * The project focuses on a relevant and modern topic: foundation models for tabular small-data learning.

2. **Both theory and practice are included**

   * This makes the repository more complete than a notebook-only benchmark repo.

3. **Multiple model families are compared**

   * The results are not limited to only TabPFN vs one baseline.

4. **Classification and regression are both covered**

   * This broadens the value of the project.

5. **Results are aggregation-ready**

   * CSV, LaTeX, Excel, PDF, and DOCX outputs show strong reporting effort.

6. **There is evidence of iterative work**

   * The commit history suggests the project matured through notebook improvements, tuning, aggregation, and reporting.

---

## What can be improved

### Weaknesses / limitations in the current repository

1. **Repository organization is still cluttered**

   * Editor folders like `.idea/` and files like `desktop.ini` should not be versioned.

2. **Naming is inconsistent**

   * Examples include spaces in file paths, duplicate filenames with `(1)`, and spelling issues like `tunning` or `3-4 Wek`.

3. **Too many generated artifacts are committed**

   * PDFs, DOCX, XLSX, HTML, copied CSVs, and duplicated files make the repo harder to navigate.

4. **Setup instructions are missing or weak**

   * A strong ML repository should clearly tell a new user how to install dependencies and rerun the analysis.

5. **Data provenance is not obvious enough**

   * Dataset sources and acquisition steps should be documented clearly.

6. **Experiment reproducibility could be improved**

   * There should be one main script or documented pipeline to reproduce all tables.

7. **Results are stronger than the repository presentation**

   * The work itself is good, but the current README does not communicate the value clearly.

---

## Recommended cleanup plan

To make the repository publication-ready, the following changes are recommended:

### 1. Clean the folder structure

Suggested layout:

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── classification/
│   └── regression/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── aggregation/
├── results/
│   ├── classification/
│   └── regression/
├── reports/
│   ├── figures/
│   └── tables/
└── presentation/
```

### 2. Add reproducibility files

Include:

* `requirements.txt`
* Python version used
* optional `environment.yml`
* dataset download instructions
* random seed policy

### 3. Separate source and generated files

Only keep the source notebooks/scripts and final key results in the repo root structure. Move heavy exports into a `reports/` or release asset section.

### 4. Add a proper `.gitignore`

Ignore:

* `.idea/`
* `__pycache__/`
* `.ipynb_checkpoints/`
* temporary docs
* OS-specific files such as `desktop.ini`

### 5. Standardize filenames

Use lowercase, underscores, and no spaces.

Example:

* `heart_dataset_summary_raw_and_processed_combined.csv`
* `classification_summary_results.csv`
* `regression_aggregated_results.csv`

---

## Suggested scientific conclusion

Based on the results present in this repository, the most defensible conclusion is:

> For small tabular datasets, TabPFN performs exceptionally well and often outperforms strong classical baselines and AutoML systems in aggregate benchmarking. It is especially attractive when the goal is to obtain strong predictive performance quickly without extensive tuning. AutoTabPFN can be competitive, but its runtime cost is much higher. CatBoost remains a strong classical competitor, while LightGBM and XGBoost offer a speed advantage but weaker aggregate performance in this project.

---

## Suggested future work

This repository can be extended in useful directions:

1. Add more datasets and report dataset metadata in one table.
2. Include statistical tests across models.
3. Add calibration plots and confusion matrices for classification.
4. Add feature importance or SHAP comparisons.
5. Include uncertainty-aware regression plots.
6. Compare raw TabPFN with TabPFN + feature engineering.
7. Benchmark against larger datasets to study where TabPFN starts to lose its advantage.
8. Package the project as a reproducible research pipeline.

---

## Final takeaway

This repository contains a **strong student/research benchmark project** with meaningful comparative results. Its main value is not just that it uses TabPFN, but that it **systematically compares TabPFN against multiple classical and AutoML baselines across both classification and regression tasks**.

The experimental results suggest that:

* **TabPFN is the strongest overall performer in this project**,
* **AutoTabPFN is competitive but much slower**,
* **CatBoost is the best traditional baseline overall**,
* **LightGBM/XGBoost remain attractive for speed**,
* and the project has enough substance to be turned into a polished academic portfolio repository with better structure and documentation.

---

## Citation

If you use this repository or build on this work, please cite the original TabPFN paper and clearly mention the datasets, preprocessing choices, and benchmark setup used in your experiments.

<img width="1162" height="648" alt="Image" src="https://github.com/user-attachments/assets/3bdd6dce-7128-4006-8fab-cea68c019445" />
<img width="1158" height="651" alt="Image" src="https://github.com/user-attachments/assets/ad4dbb15-b31b-40b3-a180-cc4de3c042cc" />
<img width="1590" height="1637" alt="Image" src="https://github.com/user-attachments/assets/c35dbbb8-256e-4ecb-8617-2a2f6aeebde8" />
<img width="1534" height="1718" alt="Image" src="https://github.com/user-attachments/assets/5440da14-203c-4225-8258-152d9cd01cce" />
<img width="1534" height="1718" alt="Image" src="https://github.com/user-attachments/assets/b356ec85-5aec-4b57-a79d-9423c2001fb6" />
