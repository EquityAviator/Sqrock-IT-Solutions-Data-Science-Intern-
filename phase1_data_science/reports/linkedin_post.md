# LinkedIn Post Draft — Phase 1 Projects

> Copy the text below, attach the screenshots from `reports/screenshots/`, and tag
> **@Sqrock IT Solutions** before publishing.

---

Excited to share my Phase 1 project work from my Data Science Internship at **Sqrock IT Solutions**! 🚀

Over the first 15 days I built **four end-to-end data science projects** on real-world datasets — going all the way from raw data to evaluated, deployable models.

**📊 Task 1 — Sales Forecasting System**
Forecast future sales from 3 million retail transactions (4.5 years of data) using Linear Regression, Decision Tree, Random Forest and ARIMA. Random Forest came out on top with an MAE of ~46,799 and RMSE of ~64,171, and I generated a 30-day forward forecast. Great practice in time-series feature engineering — lag features, rolling means and seasonality.

**🎓 Task 2 — Student Performance Predictor**
Predicted students' final marks from study hours, attendance and prior grades using the UCI Student Performance dataset. Random Forest reached an **R² of 0.831** with an average error under 1 mark — the clearest lesson was that prior attainment (not study time alone) dominates performance.

**📄 Task 3 — AI Resume Screening System**
Built an NLP pipeline that screens 825 resumes against a job description: text cleaning → lemmatisation → TF-IDF → cosine similarity + skill matching → ranked shortlist with match percentages. Also handled the full PDF-to-text extraction flow.

**📉 Task 4 — Customer Churn Prediction System**
Predicted telecom churn across 7,043 customers using Logistic Regression, Decision Tree, Random Forest and KNN. Best model: **F1 0.619, ROC-AUC 0.835**. Contract type and tenure emerged as the strongest churn drivers — and I made sure the risk calculator outputs calibrated probabilities, not just labels.

**🛠️ Technologies used**
Python · Pandas · NumPy · Scikit-learn · Statsmodels (ARIMA) · NLTK · Matplotlib · Seaborn · pypdf · Chart.js

**📚 What I learned**
- Why a strict time-based split matters for forecasting (and how easy it is to leak future data)
- Handling class imbalance — and why balanced weights inflate probabilities, so calibration matters
- That clean NLP preprocessing does more for retrieval quality than a fancier model
- Turning analysis into something a non-technical stakeholder can actually use through an interactive dashboard

I also built an interactive dashboard that explains each project and lets you run the trained models live (student-mark predictor, churn-risk calculator and a searchable resume ranking table) — link in the comments. 💡

Big thanks to **Sqrock IT Solutions** for the structured, hands-on internship format. On to Phase 2! 🙌

#DataScience #MachineLearning #Python #NLP #DataAnalytics #Internship #ScikitLearn #DeepLearning #SqrockITSolutions

---

## Screenshots to attach

| # | File | Shows |
|---|------|-------|
| 1 | `reports/screenshots/dashboard_overview.png` | Phase 1 overview, task catalogue and dataset provenance |
| 2 | `reports/screenshots/dashboard_task1.png` | Sales forecasting EDA + actual vs predicted + 30-day forecast |
| 3 | `reports/screenshots/dashboard_task2.png` | Student performance EDA + model metrics + live predictor |
| 4 | `reports/screenshots/dashboard_task3.png` | Resume ranking, score distribution and skill coverage |
| 5 | `reports/screenshots/dashboard_task4.png` | Churn EDA, confusion matrix, ROC curves + risk calculator |

## Checklist before posting

- [ ] Attach at least 3–5 screenshots (the brief requires demo images)
- [ ] Tag **@Sqrock IT Solutions** in the post body
- [ ] Add the live dashboard link in the first comment
- [ ] Add the GitHub repository link in the first comment
- [ ] Proofread names and numbers
