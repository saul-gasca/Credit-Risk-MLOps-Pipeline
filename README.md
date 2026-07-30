# Credit Risk Predictor (Loan Default Prediction)

An end-to-end MLOps pipeline for predicting loan default risk, built on the **Home Credit Default Risk** dataset (Kaggle). The project goes beyond a single flat table, incorporating multi-source relational data (application, bureau, bureau balance, previous applications, installment payments) through custom feature engineering, with rigorous model comparison, hyperparameter tuning, and explainability.

## Project Goal

Predict whether a loan applicant will default (`TARGET = 1`) or repay successfully (`TARGET = 0`), using not only the applicant's current application data but also their historical behavior with external credit bureaus and with Home Credit itself.

## Dataset

- **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
- **Tables used:**
  - `application_train.csv` — main table, one row per client (307,511 rows, 122 columns), contains `TARGET`
  - `bureau.csv` — client's credit history at other financial institutions (1-to-many with client)
  - `bureau_balance.csv` — monthly balance/status history for each bureau credit (indirect relation via `bureau`)
  - `previous_application.csv` — client's previous loan applications with Home Credit (1-to-many with client)
  - `installments_payments.csv` — actual vs. expected installment payments (indirect relation via `previous_application`)
- **Class balance:** ~91.9% non-default vs. ~8.1% default — significant class imbalance

## Project Structure

```
credit-risk-mlops/
├── datasets/
│   ├── raw/                  # original Kaggle CSVs (DVC-tracked, git-ignored)
│   └── processed/            # merged + cleaned + encoded dataset (DVC-tracked)
├── notebooks/
│   ├── 01_eda_application_train.ipynb
│   ├── 02_eda_bureau.ipynb
│   ├── 03_eda_bureau_balance.ipynb
│   ├── 04_eda_previous_application.ipynb
│   ├── 05_eda_installments_payments.ipynb
│   └── 06_feature_engineering_merge.ipynb
├── scripts/
│   └── download_data.sh      # automated Kaggle data download
├── images/                   # saved plots (EDA, ROC/PR curves, SHAP)
├── .dvc/
├── .gitignore
└── README.md
```

## Pipeline Overview

### 1. Data Ingestion
- Automated download of all 5 raw tables from Kaggle via a bash script (`scripts/download_data.sh`), using the Kaggle API.
- Raw data versioned with **DVC**, keeping the Git repository lightweight while still tracking dataset versions.

### 2. Exploratory Data Analysis (EDA)
Each of the 5 tables was explored independently (one notebook per table) before any merging, covering:
- Shape, key uniqueness, and table granularity (1-to-1 vs 1-to-many relationships)
- Missing value analysis
- Categorical cardinality
- Detection of sentinel/placeholder values (e.g., `DAYS_EMPLOYED` containing `365243` as a "not applicable" code for retired/unemployed applicants, later confirmed to appear across multiple `DAYS_*` columns in `previous_application` as well)
- Duplicate validation against the correct composite key for each table (including a case in `installments_payments` where apparent duplicates were confirmed to be legitimate partial payments, not data errors)

### 3. Feature Engineering (Multi-Table Aggregation)
Since only `application_train` is at client-level (`SK_ID_CURR`), the other 4 tables required aggregation before merging:

- **`bureau` → client level:** aggregated directly by `SK_ID_CURR` (count, mean, max, min, sum of credit amounts, overdue days, etc.)
- **`bureau_balance` → `bureau` → client level (double join):** monthly status history aggregated by `SK_ID_BUREAU` first (including a custom `STATUS_SEVERITY` mapping to convert categorical payment status into an ordinal severity scale), merged into `bureau`, then re-aggregated to client level
- **`previous_application` → client level:** aggregated directly by `SK_ID_CURR`
- **`installments_payments` → `previous_application` → client level (double join):** custom features engineered (`DAYS_LATE`, `AMT_SHORTFALL`) to capture payment delinquency and payment shortfall, aggregated by `SK_ID_PREV` first, merged into `previous_application`, then re-aggregated to client level

Final merge: all aggregated tables joined to `application_train` via `SK_ID_CURR` using **left joins** (to preserve all 307,511 original clients, even those without bureau or previous application history).

### 4. Data Cleaning & Encoding
- **Missing value threshold:** columns with >60% missing values dropped (consistently applied across all tables)
- **Sentinel value handling:** `DAYS_EMPLOYED` anomaly flagged (`DAYS_EMPLOYED_ANOMALY` boolean) before replacing the placeholder with `NaN`
- **Differentiated null handling:** count/sum aggregates from client with no bureau/previous-application history filled with `0` (a true zero, not a missing value); mean/max/min aggregates left as `NaN`, since LightGBM/XGBoost handle missing values natively without requiring an artificial imputed value
- **Encoding:**
  - Low-cardinality categoricals (≤8 categories) → one-hot encoding (`pd.get_dummies`, `drop_first=True`)
  - High-cardinality categoricals (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) → frequency encoding, to avoid excessive dimensionality from one-hot encoding

Final processed dataset: **307,511 rows × 165 columns**, fully numeric/boolean, DVC-tracked.

### 5. Modeling

**Train/validation split:** 80/20, stratified on `TARGET` to preserve class balance in both sets.

**Model comparison (5-fold Stratified Cross-Validation on training set):**

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline, median-imputed) | 0.1409 | 0.1409 | 0.6610 |
| LightGBM (default params, `class_weight='balanced'`) | 0.2519 | 0.2639 | 0.7726 |
| XGBoost (default params, `scale_pos_weight`) | 0.2556 | 0.2683 | 0.7730 |
| LightGBM (Optuna-tuned, 20 trials) | 0.2602 | — | — |
| **XGBoost (Optuna-tuned, 20 trials) — final model** | **0.2598** | **0.2730** | **0.7770** |

**Hyperparameter tuning:** performed with **Optuna** (Tree-structured Parzen Estimator sampler), optimizing directly for PR-AUC via 5-fold cross-validation, tuning `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and `min_child_weight`.

**Experiment tracking:** all runs (parameters, metrics, and serialized models) logged with **MLflow**.

### 6. Model Selection: Why XGBoost

LightGBM and XGBoost performed almost identically both before and after tuning (a ~0.0004 PR-AUC difference after tuning, well within cross-validation noise). Since neither model showed a decisive statistical advantage, **XGBoost was selected as the final production model** based on:
- Marginally higher Test PR-AUC and ROC-AUC after tuning
- Equivalent training/inference cost for this dataset size
- To keep a single, coherent narrative across tuning (Optuna) and explainability (SHAP) rather than splitting effort across two equally-performing models

This decision is a practical one, not a claim that XGBoost is categorically superior to LightGBM — with these features and this amount of data, both gradient boosting implementations reach a very similar performance ceiling.

### 7. Explainability (SHAP)

`TreeExplainer` was used on the final XGBoost model to interpret predictions:
- **Global importance (summary plot & bar plot):** `EXT_SOURCE_3`, `EXT_SOURCE_2`, and `EXT_SOURCE_1` (external bureau scores) dominate feature importance, followed by `AMT_GOODS_PRICE`, `AMT_CREDIT`, and several engineered features from the bureau/previous-application aggregations (`CNT_PAYMENT_MAX`, `DAYS_LATE_MAX_MAX`, `AMT_CREDIT_SUM_DEBT_MEAN`, `AMT_SHORTFALL_MEAN_MEAN`) — confirming that the multi-table feature engineering contributes real, non-trivial predictive signal, not just noise.
- **Local explanations (force plots):** individual client predictions decomposed into per-feature contributions, useful for justifying/explaining individual credit decisions — directly relevant to regulatory explainability requirements in lending.

## Why `average_precision_score` (PR-AUC) as the Primary Metric

This was a deliberate choice driven by the severe class imbalance (~8% positive class), and it's worth explaining why it was preferred over ROC-AUC, F1, or a raw confusion matrix:

- **`average_precision_score` (PR-AUC)** summarizes the precision-recall curve across *all* possible decision thresholds, without committing to any single cutoff. With only ~8% positives, this metric is far more sensitive to how well the model ranks the minority (default) class — it does not get inflated by the large number of true negatives the way ROC-AUC can. A random classifier's PR-AUC baseline equals the positive class rate (~0.08), not 0.5, making it a much stricter benchmark for imbalanced problems.
- **ROC-AUC** was still tracked as a secondary/complementary metric (it's a very standard, widely understood metric), but on its own it can look "deceptively good" on imbalanced data, since the false positive rate is diluted by the large majority class.
- **F1-score and the confusion matrix** both require choosing a fixed decision threshold (default 0.5) to convert probabilities into class labels. With this level of imbalance, 0.5 is almost never the optimal threshold — the model needs a business-driven threshold decision first (e.g., optimizing for a specific cost of false negatives vs. false positives), which was intentionally left as a later step rather than baked into early model comparison. Using threshold-dependent metrics to *compare* candidate models before a threshold is chosen would have made the comparison unstable and threshold-dependent rather than a fair, ranking-based comparison of the models themselves.

In short: PR-AUC and ROC-AUC were used to **compare and select the model** (threshold-independent, ranking-based), while F1/confusion matrix are reserved for a later step — evaluating the *final* model's behavior once a specific operating threshold is chosen for deployment.

## Tech Stack

Python, pandas, scikit-learn, LightGBM, XGBoost, Optuna, SHAP, MLflow, DVC, Git

## Roadmap (Next Phases)

- [ ] Threshold optimization for the final model (cost-based analysis, similar to prior Telco Churn project)
- [ ] Apache Airflow DAG for orchestrated data ingestion and feature pipeline
- [ ] BigQuery integration as the data warehouse layer
- [ ] Model serving via FastAPI + Docker
- [ ] CI/CD with GitHub Actions
- [ ] Data/model drift monitoring with Evidently AI

## Author

Saul Gasca Farrera — [GitHub](https://github.com/saul-gasca)
