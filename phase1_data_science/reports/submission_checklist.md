# Final Submission Checklist — Phase 1

Based on the *Final Submission Instructions* in the Phase 1 brief.

---

## A. Project completion (brief requirement: any 3 of 4)

| Task | Status | Best model | Headline metric |
|------|--------|-----------|-----------------|
| 1 — Sales Forecasting System | ✅ Complete | Random Forest | MAE 46,799 · RMSE 64,171 |
| 2 — Student Performance Predictor | ✅ Complete | Random Forest | R² 0.831 · MAE 0.901 |
| 3 — AI Resume Screening System | ✅ Complete | TF-IDF + cosine similarity | 825 ranked · 10 shortlisted |
| 4 — Customer Churn Prediction System | ✅ Complete | Random Forest | F1 0.619 · AUC 0.835 |

**Result: 4 / 4 completed — exceeds the minimum of 3.**

---

## B. Brief's mandatory submission steps

### 1. Upload all project source code on GitHub

- [ ] Create a GitHub repository (e.g. `data-science-internship-phase1`)
- [ ] Push the entire `phase1_internship/` folder
- [ ] Confirm `README.md` renders correctly on the repo home page
- [ ] Add a `.gitignore` that excludes `data/` (large files — re-fetchable via `download_data.py`)
- [ ] Verify each task folder contains both the script and its `artifacts/` folder

```bash
cd phase1_internship
git init
git add .
git commit -m "Phase 1: four data science projects with dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

### 2. Deploy / host the project and submit the live link

The dashboard in `web/` is a static site, so deployment is straightforward.

- [ ] **GitHub Pages** — push the repo, then Settings → Pages → deploy from `/web`
- [ ] or **Netlify** — drag-and-drop the `web/` folder at app.netlify.com/drop
- [ ] or **Vercel** — `vercel deploy` inside the `web/` folder
- [ ] Verify the live URL loads and all charts render
- [ ] Record the live URL: `_______________________________`

> Note: the dashboard is fully self-contained (Chart.js is vendored locally and data is bundled into
> `assets/data.js`), so no backend or CDN is required.

### 3. Create a professional LinkedIn post

- [ ] Use the draft in `reports/linkedin_post.md`
- [ ] Explain the project, the technologies used and the learning experience
- [ ] Attach screenshots / demo images
- [ ] **Tag @Sqrock IT Solutions**
- [ ] Add the live link and GitHub link in the first comment
- [ ] Publish and save the post URL: `_______________________________`

### 4. Include project screenshots or demo images

- [ ] Attach at least 3–5 images from `reports/screenshots/`
- [ ] Recommended set: `dashboard_overview`, `dashboard_task1`, `dashboard_task2`,
      `dashboard_task3`, `dashboard_task4`

---

## C. Quality verification (done)

| Check | Result |
|-------|--------|
| Every required pipeline stage implemented per task | ✅ (see `reports/Phase1_Report.md` §1.3, §3–6) |
| All required evaluation metrics reported | ✅ MAE/RMSE · MAE/MSE/R² · match % · Acc/Prec/Rec/F1/CM |
| No data leakage (time-based split for forecasting, stratified split for churn) | ✅ |
| Datasets sourced from reputable public repositories with rationale documented | ✅ (4 datasets, see README) |
| All scripts run end-to-end without errors | ✅ |
| Dashboard renders with no JavaScript console errors | ✅ |
| Reproducible from a clean clone | ✅ (`download_data.py` + 4 scripts) |

---

## D. File manifest

```
phase1_internship/
├── README.md                        # project overview & quick start
├── requirements.txt                 # python dependencies
├── download_data.py                 # reproducible dataset fetcher (with provenance)
├── data/                            # datasets (re-fetchable; excluded from git)
├── task1_sales_forecasting/         # sales_forecasting.py + artifacts/ (metrics, CSVs, 6 charts)
├── task2_student_performance/       # student_performance.py + artifacts/ (metrics, CSVs, 7 charts)
├── task3_resume_screening/          # resume_screening.py + artifacts/ (ranking, shortlist, 3 charts)
├── task4_churn_prediction/          # churn_prediction.py + artifacts/ (metrics, CSVs, 8 charts)
├── web/                             # interactive dashboard (index.html, styles.css, app.js, assets/)
└── reports/
    ├── Phase1_Report.md             # full requirements analysis & execution report
    ├── linkedin_post.md             # LinkedIn post draft + attachment checklist
    ├── submission_checklist.md      # this file
    └── screenshots/                 # 6 dashboard screenshots for the LinkedIn post
```
