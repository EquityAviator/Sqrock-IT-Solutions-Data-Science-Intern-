# Phase 1 — Requirements Analysis & Execution Report

**Programme:** Data Science Internship — Sqrock IT Solutions
**Phase:** Phase 1 (First 15 Days)
**Scope delivered:** All 4 projects completed (requirement: any 3)
**Report date:** 12 October 2026

---

## 1. Analysis of Phase 1 requirements

The Phase 1 brief (`_Data_Science.pdf`) defines a 15-day project phase in which interns are given
**four project tasks** and must complete **any three** successfully within the deadline. The stated
purpose is to improve practical development skills, project-building experience, problem-solving
ability and portfolio quality through real-world projects.

### 1.1 Objectives

| # | Objective | How it was met |
|---|-----------|----------------|
| 1 | Build practical, real-world data science projects | 4 full projects built on real datasets, not toy data |
| 2 | Follow a complete project lifecycle | Every project runs collect → clean → EDA → features → model → predict → evaluate |
| 3 | Demonstrate problem-solving ability | Model selection, leakage avoidance, class imbalance handling, probability calibration |
| 4 | Produce portfolio-quality output | Source code, artifacts, interactive dashboard, screenshots, LinkedIn draft |

### 1.2 The four tasks and their required deliverables

| Task | Title | Type | Required evaluation metrics |
|------|-------|------|-----------------------------|
| 1 | Sales Forecasting System | Time-series regression | MAE, RMSE |
| 2 | Student Performance Predictor | Regression | MAE, MSE, R² |
| 3 | AI Resume Screening System | NLP / ranking | Match score (%), ranking list, shortlist |
| 4 | Customer Churn Prediction System | Classification | Accuracy, Precision, Recall, F1, Confusion Matrix |

### 1.3 Mandatory pipeline stages (specified per task)

1. **Data Collection** — CSV dataset from Kaggle / sample dataset / created data
2. **Data Cleaning / Preprocessing** — missing values, type conversion, de-duplication, encoding, scaling, train/test split
3. **Exploratory Data Analysis** — trends, distributions, correlations, seasonality, visualisations (line/bar/histogram/scatter/heatmap/pie)
4. **Feature Engineering** — derived and time-based features
5. **Model Building** — the specific algorithms named per task
6. **Prediction** — forward forecasts / predicted-vs-actual comparisons
7. **Evaluation** — the exact metrics listed above

### 1.4 Final submission requirements

The brief lists four mandatory submission steps after completing the projects:

1. Upload all project source code on **GitHub**
2. **Deploy / host** the project online and submit the live project link
3. Create a professional **LinkedIn post** explaining the project, technologies used and learning experience
4. Include project **screenshots / demo images** and **tag Sqrock IT Solutions** in the post

### 1.5 Deadlines & evaluation criteria

- **Deadline:** the first 15 days of the internship programme (Phase 1 window).
- **Pass threshold:** at least 3 of 4 projects completed successfully.
- **Quality criteria:** correctness of the pipeline, appropriateness of models, and validity of the
  reported evaluation metrics.

---

## 2. Dataset sourcing — decisions and rationale

Where the brief allowed a choice of dataset, a **real, reputable, publicly available** dataset was
preferred over synthetic data so that the reported metrics reflect genuine difficulty.

| Task | Dataset chosen | Source | Size | Rationale |
|------|----------------|--------|------|-----------|
| 1 | Store Sales (Favorita grocery chain) | Hugging Face mirror of the Kaggle *Store Sales – Time Series Forecasting* dataset | 3,000,888 rows · 1,688 daily points · 2013-01-01 → 2017-08-15 | Provides `date` and `sales` (required) plus `family` (category) and `store_nbr` (region/store) — every optional column in the brief. Four and a half years of real retail transactions give genuine trend, weekly and yearly seasonality, making MAE/RMSE meaningful. |
| 2 | Student Performance (Cortez & Silva) | UCI Machine Learning Repository | 1,044 students (Math 395 + Portuguese 649) | The canonical, peer-reviewed student-performance dataset. Its variables map cleanly onto every column the brief requests: `studytime` → study hours, `absences` → attendance, `G2` → previous score, `G1` → assignment score, `G3` → final marks (target), plus gender and parent education. |
| 3 | Resume Screening Dataset | Hugging Face (`AzharAli05/Resume-Screening-Dataset`) | 10,174 resumes · 45 roles | Contains full unstructured resume text with candidate names, matching job descriptions and human select/reject decisions — ideal for TF-IDF / cosine-similarity screening and sanity-checking the ranking. |
| 4 | Telco Customer Churn | IBM sample dataset (mirrored on GitHub / Hugging Face) | 7,043 customers · 21 columns | The industry-standard churn benchmark. Includes demographics, subscription plan, `tenure`, `MonthlyCharges`, `TotalCharges`, service usage and a clean Yes/No `Churn` label, with a realistic ~26.5% churn rate that makes Precision/Recall/F1 informative. |

All sources are re-fetchable via `download_data.py`, which records the URL, source and rationale.

---

## 3. Task 1 — Sales Forecasting System

### 3.1 Steps taken

| Stage | What was done |
|-------|---------------|
| 1. Data Collection | Loaded 3,000,888 transaction rows (`date`, `store_nbr`, `family`, `sales`, `onpromotion`). |
| 2. Data Cleaning | Parsed `date` to datetime, dropped null/duplicate rows, aggregated to a daily total series, re-indexed onto a complete calendar (1,688 days) and time-interpolated 4 missing days. |
| 3. EDA | Sales trend line (with 30-day rolling mean); monthly totals with best/worst month highlighted; calendar-month seasonality; day-of-week pattern. |
| 4. Feature Engineering | Year, month, day, day-of-week, day-of-year, week-of-year, weekend flag, promotion count, plus lags (1, 7, 14, 30 days) and rolling means (7, 30 days). |
| 5. Model Building | Linear Regression, Decision Tree Regressor, Random Forest Regressor, and ARIMA(7,1,1) as the optional advanced time-series model. |
| 6. Prediction | Time-based hold-out of the final 90 days for evaluation; recursive 30-day forward forecast using the best model. |
| 7. Evaluation | MAE and RMSE on the hold-out window. |

### 3.2 Results

| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 62,115.72 | 82,427.69 |
| Decision Tree | 71,738.65 | 104,803.82 |
| **Random Forest** ⭐ | **46,799.08** | **64,170.54** |
| ARIMA(7,1,1) | 119,375.41 | 148,932.60 |

**Best model: Random Forest** (lowest RMSE). Random Forest outperforms plain linear regression by
capturing the non-linear weekday/weekend and holiday effects that a single linear trend cannot.

**Key EDA findings**
- Sales grew substantially over the period; the **highest month was 2016-12** (≈30.8M) and the
  **lowest was 2013-02** (≈9.7M) — a mix of underlying growth and Christmas seasonality.
- Clear weekly seasonality with elevated weekend sales, and a strong December peak.
- **30-day forward forecast:** ≈23,539,632 total (≈784,654 per day).

### 3.3 Artifacts
`metrics.json`, `forecast_30d.csv`, `actual_vs_predicted.csv`, `monthly_totals.csv` +
6 charts (`t1_trend`, `t1_monthly`, `t1_seasonality`, `t1_dow`, `t1_actual_vs_pred`, `t1_forecast`).

---

## 4. Task 2 — Student Performance Predictor

### 4.1 Steps taken

| Stage | What was done |
|-------|---------------|
| 1. Data Collection | Combined the UCI Math and Portuguese cohorts (1,044 students). |
| 2. Preprocessing | No missing values found; mapped the UCI variables onto the brief's schema (studytime → hours/week, absences → attendance %, G1/G2/G3 → assignment/previous/final marks); label-encoded 18 categorical variables; standard-scaled; 80/20 train/test split (835 / 209). |
| 3. EDA | Correlation heatmap; study-hours vs final-marks scatter with trend line; average marks by attendance band; distribution of final marks; per-feature correlation with the target. |
| 4. Model Building | Linear Regression, Decision Tree Regressor (depth 6), Random Forest Regressor (400 trees). |
| 5. Train & Test | Trained on 835 students, evaluated on 209 unseen students. |
| 6. Evaluation | MAE, MSE and R². |

### 4.2 Results

| Model | MAE | MSE | R² |
|-------|-----|-----|-----|
| Linear Regression | 0.969 | 2.936 | 0.810 |
| Decision Tree | 0.991 | 3.307 | 0.786 |
| **Random Forest** ⭐ | **0.901** | **2.609** | **0.831** |

**Best model: Random Forest** — explains **83.1%** of the variance in final marks, with an average
error of under 1 mark on a 0–20 scale.

**Key EDA findings**
- `previous_score` (G2) and `assignments_score` (G1) are by far the strongest predictors — prior
  attainment dominates.
- Study hours show only a weak positive correlation with final marks (r ≈ 0.16) on their own, but
  contribute usefully alongside prior grades.
- Past failures correlate negatively with final marks; the distribution of final marks is left-skewed
  with a mean of 11.34/20.

### 4.3 Artifacts
`metrics.json`, `predictions.csv`, `sample_students.csv`, `simple_model.json` +
7 charts (correlation heatmap, study vs marks, attendance bands, score distribution, feature
correlation, feature importance, actual vs predicted).

---

## 5. Task 3 — AI Resume Screening System

### 5.1 Steps taken

| Stage | What was done |
|-------|---------------|
| 1. Data Collection | Loaded 10,174 resume records across 45 roles; normalised role labels and filtered to **Data Scientist** (825 unique resumes). Candidate names extracted from the CV text. **PDF round-trip demo:** 3 resumes rendered to PDF and re-parsed with `pypdf` (avg 2,586 chars) to satisfy the "convert PDFs into text" requirement. |
| 2. NLP Preprocessing | Lowercase → strip URLs, emails, punctuation and digits → tokenise → remove stopwords (extended with CV boilerplate) → WordNet lemmatisation. |
| 3. Feature Extraction | TF-IDF vectoriser (1–2 grams, 3,000 features, sublinear TF). Bag-of-Words computed in parallel for comparison. |
| 4. Matching Algorithm | Cosine similarity between each resume vector and the job-description vector (65% weight) blended with skill/keyword coverage against a 31-term skill lexicon (35% weight). |
| 5. Ranking System | Sorted by blended score, rescaled to a 0–100% match percentage, top 10 shortlisted above a 30% cut-off. |
| 6. Output | Candidate name, match %, full ranking list and shortlist exported to CSV. |

### 5.2 Results

- **825 resumes screened** for the *Data Scientist* role; **10 shortlisted**.
- **Top candidate: Malay Gulati — 100% match**, with 23 of 31 target skills present (python, sql,
  machine learning, deep learning, nlp, tensorflow, spark, aws, azure, time series, a/b testing, …).
- Mean match score across the pool: 62.6%.

**Skill-matching fix.** An earlier version matched skills by plain substring, so the skill "r" was
found in every resume, "git" inside "digital", "aws" inside "laws", and so on. Skills are now matched
as whole words/phrases (with "sklearn" accepted for scikit-learn and "A/B testing" variants handled).

**Validation note.** The dataset ships human select/reject labels, but they are noisy — resumes are
frequently rejected for reasons unrelated to skill match (e.g. "lacked leadership skills for a senior
position"). With the fixed skill matching, the shortlist's select-rate is **0.60 vs a pool baseline of
0.498 (+10 points)**; before the fix it was 0.50, i.e. no better than chance. With only 10 shortlisted
resumes this is an indicative signal, not a precision score.

### 5.3 Artifacts
`metrics.json`, `ranking.csv`, `shortlist.csv` +
3 charts (top candidates, score distribution, skill coverage) + 3 sample PDF resumes.

---

## 6. Task 4 — Customer Churn Prediction System

### 6.1 Steps taken

| Stage | What was done |
|-------|---------------|
| 1. Data Collection | Loaded 7,043 IBM Telco customers with 21 attributes. |
| 2. Preprocessing | Coerced `TotalCharges` to numeric (11 blank strings imputed from `MonthlyCharges`), removed duplicates, label-encoded 16 categorical columns, standard-scaled, stratified 80/20 split (5,634 / 1,409). |
| 3. EDA | Churn distribution (counts + donut); monthly charges vs churn; churn rate by tenure band; correlation heatmap. |
| 4. Feature Engineering | Created `avg_monthly_spend`, `contract_group`, `tenure_bucket` and `num_services`; dropped the irrelevant `CustomerID` as the brief requires. |
| 5. Model Building | Logistic Regression, Decision Tree, Random Forest and KNN — all class-balanced to handle the 26.5% minority class. |
| 6. Evaluation | Accuracy, Precision, Recall, F1, confusion matrix and ROC-AUC. |

### 6.2 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.739 | 0.505 | 0.791 | 0.617 | 0.839 |
| Decision Tree | 0.714 | 0.477 | 0.808 | 0.600 | 0.817 |
| **Random Forest** ⭐ | **0.764** | 0.541 | 0.722 | **0.619** | 0.835 |
| KNN | 0.774 | 0.580 | 0.535 | 0.556 | 0.809 |

**Best model: Random Forest** by F1. Confusion matrix (test set): TN 806, FP 229, FN 104, TP 270.

**Key EDA findings**
- Overall churn rate **26.5%** — a moderate class imbalance.
- **Contract type is the single strongest churn driver**, followed by tenure, online security and
  tech support. Month-to-month customers churn far more than annual or two-year customers.
- Churn is concentrated among **new customers** (first year) and those on **higher monthly charges**,
  particularly fibre-optic subscribers.

**Calibration.** The interactive calculator uses an *unweighted* logistic model so its probabilities
are calibrated — mean predicted probability 0.266 against an actual churn rate of 0.265. The balanced
variants used for the F1 comparison deliberately trade precision for recall.

### 6.3 Artifacts
`metrics.json`, `at_risk_customers.csv`, `logit_model.json` +
8 charts (churn distribution, charges, tenure, correlation heatmap, confusion matrix, model
comparison, ROC curves, feature importance).

---

## 7. Interactive dashboard (representation & explanation)

A single-page dashboard was built to present every project with both explanation and live
demonstration: **`web/index.html`**.

- **Overview page** — Phase 1 requirements analysis, task catalogue with headline metrics, a
  pipeline-mapping matrix showing how each project satisfies all seven required stages, and the
  dataset provenance table.
- **Per-task pages** — problem statement, dataset source and rationale, step-by-step pipeline,
  interactive charts (Chart.js) and the original EDA figures.
- **Live demos** — a student-mark predictor driven by the exported linear coefficients (Task 2), a
  churn-risk calculator running the exported calibrated logistic model entirely in the browser
  (Task 4), and a searchable / filterable resume ranking table (Task 3).
- **Submission page** — the delivery checklist and reproduction instructions.

The dashboard is fully offline-capable: Chart.js is vendored locally and all data is bundled into
`assets/data.js`, so `web/index.html` can be opened directly or deployed as a static site.

---

## 8. Deliverables summary

| Deliverable | Location | Status |
|-------------|----------|--------|
| Task 1 source + artifacts | `task1_sales_forecasting/` | ✅ Complete |
| Task 2 source + artifacts | `task2_student_performance/` | ✅ Complete |
| Task 3 source + artifacts | `task3_resume_screening/` | ✅ Complete |
| Task 4 source + artifacts | `task4_churn_prediction/` | ✅ Complete |
| Reproducible dataset fetcher | `download_data.py` | ✅ Complete |
| Interactive dashboard | `web/` | ✅ Complete |
| Dashboard screenshots | `reports/screenshots/` | ✅ Complete |
| Phase 1 report (this document) | `reports/Phase1_Report.md` | ✅ Complete |
| LinkedIn post draft | `reports/linkedin_post.md` | ✅ Complete |
| Submission checklist | `reports/submission_checklist.md` | ✅ Complete |

---

## 9. Remaining manual steps (outside automation scope)

1. **Push to GitHub** — create a repository and push the `phase1_internship/` folder.
2. **Deploy the dashboard** — publish the `web/` folder to GitHub Pages / Netlify / Vercel and record
   the live URL.
3. **Publish the LinkedIn post** — use `reports/linkedin_post.md`, attach the screenshots from
   `reports/screenshots/`, and tag **@Sqrock IT Solutions**.

---

*Prepared as part of the Sqrock IT Solutions Data Science Internship — Phase 1 (First 15 Days).*
