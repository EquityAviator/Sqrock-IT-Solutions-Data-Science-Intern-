# Sqrock IT Solutions — Data Science Internship — Phase 1

**First 15 Days · 4 project tasks · all 4 completed**

This repository contains the complete Phase 1 deliverable: four end-to-end data science
projects built on real, publicly-sourced datasets, plus an interactive dashboard that
explains each project and demonstrates the trained models.

---

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. download every dataset (with provenance)
python download_data.py

# 3. run each project (each writes charts + metrics + CSVs into its own artifacts/ folder)
python task1_sales_forecasting/sales_forecasting.py
python task2_student_performance/student_performance.py
python task3_resume_screening/resume_screening.py
python task4_churn_prediction/churn_prediction.py

# 3b. after re-running any task, rebundle the dashboard data and (optionally) the notebooks
python web/build_data.py
python tools/make_notebooks.py

# 4. open the dashboard
#    simply open web/index.html in a browser, or serve it:
python -m http.server 8000 --directory web
```

---

## Projects & results

| # | Project | Type | Best model | Headline metric |
|---|---------|------|-----------|-----------------|
| 1 | Sales Forecasting System | Time series / regression | Random Forest | **MAE 46,799 · RMSE 64,171** |
| 2 | Student Performance Predictor | Regression | Random Forest | **R² 0.831 · MAE 0.901** |
| 3 | AI Resume Screening System | NLP / ranking | TF-IDF + cosine similarity | **825 resumes ranked, 10 shortlisted** |
| 4 | Customer Churn Prediction | Classification | Random Forest | **F1 0.619 · AUC 0.835 · Acc 0.764** |

---

## Datasets (source + rationale)

| Task | Dataset | Source | Size | Why |
|------|---------|--------|------|-----|
| 1 | Store Sales (Favorita grocery) | Hugging Face (Kaggle mirror) | 3,000,888 rows | Real 4.5-year multi-store retail log — genuine trend, weekly and yearly seasonality instead of a synthetic toy series. |
| 2 | Student Performance (Cortez & Silva) | UCI ML Repository | 1,044 students | The canonical peer-reviewed student dataset; contains every factor the brief lists (study time, absences, prior grades, parent education). |
| 3 | Resume Screening Dataset | Hugging Face (AzharAli05) | 10,174 resumes / 45 roles | Full unstructured CV text with candidate names, job descriptions and human select/reject decisions. |
| 4 | Telco Customer Churn | IBM sample dataset | 7,043 customers | Industry-standard churn benchmark with real class imbalance (~26% churn). |

All datasets are fetched by `download_data.py`, which records the URL, source and rationale for each.

---

## Jupyter notebooks

Each task folder also has an executed `.ipynb` with the same pipeline cell by cell, every chart
shown inline. They are generated from the `.py` scripts by `tools/make_notebooks.py`, so the
script stays the single source of truth.

## Repository layout

```
phase1_internship/
├── download_data.py                  # reproducible dataset fetcher (with provenance)
├── requirements.txt
├── data/                             # datasets (not committed — re-fetch with download_data.py)
├── task1_sales_forecasting/
│   ├── sales_forecasting.py
│   └── artifacts/                    # metrics.json, forecast_30d.csv, actual_vs_predicted.csv, charts
├── task2_student_performance/
│   ├── student_performance.py
│   └── artifacts/                    # metrics.json, predictions.csv, simple_model.json, charts
├── task3_resume_screening/
│   ├── resume_screening.py
│   └── artifacts/                    # ranking.csv, shortlist.csv, metrics.json, charts
├── task4_churn_prediction/
│   ├── churn_prediction.py
│   └── artifacts/                    # at_risk_customers.csv, logit_model.json, metrics.json, charts
├── web/                              # interactive dashboard (index.html, styles.css, app.js, assets/)
└── reports/                          # Phase 1 report, LinkedIn post, submission checklist, screenshots
```

---

## Interactive dashboard

`web/index.html` is a self-contained static dashboard (works offline — Chart.js is vendored
locally and all data is bundled into `assets/data.js`). It includes:

- **Overview** — Phase 1 requirements analysis, task catalogue, pipeline mapping, dataset provenance
- **Per-task pages** — problem statement, dataset rationale, step-by-step pipeline, interactive charts
- **Live demos** — a student-mark predictor (Task 2) and a churn-risk calculator (Task 4) that run the
  exported model coefficients in the browser, plus a searchable/filterable resume ranking table (Task 3)
- **Submission** — delivery checklist and reproduction instructions

Deploy the `web/` folder to GitHub Pages, Netlify or Vercel for the required live link.

---

## Notes on methodology

- **No data leakage.** Task 1 uses a strict time-based split (train on all but the last 90 days);
  Task 4 uses a stratified 80/20 split.
- **Honest metrics.** Real datasets are used throughout, so scores reflect genuine difficulty rather
  than a trivially-separable synthetic set.
- **Calibrated probabilities.** The Task 4 churn calculator uses an *unweighted* logistic model so the
  probabilities are calibrated (mean predicted 0.266 vs actual churn rate 0.265); the balanced variants
  used for F1 comparison deliberately trade precision for recall.
- **Task 3 validation.** The shortlist is compared against the dataset's own labels as an indicative
  signal only — those labels are noisy (resumes are frequently rejected for reasons unrelated to
  skill match).
