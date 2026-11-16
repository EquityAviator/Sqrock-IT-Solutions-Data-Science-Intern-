# LinkedIn post (draft)

Phase 2 of my Data Science internship at @Sqrock IT Solutions is complete: four projects, all on real public data, in one dashboard.

**1. Stock Market Prediction** – 10 US stocks from Yahoo Finance (2015–2026). Linear Regression, Random Forest, XGBoost and an LSTM trained on my GPU. R² 0.988 and 1.5% price error, and an honest finding: that only matches the "tomorrow = today" baseline. The real edge is 53% on direction.

**2. AI Data Analysis Dashboard** – upload any CSV or Excel file; it cleans the data, profiles it and writes ranked insights with recommendations using significance tests and an Isolation Forest.

**3. Smart Inventory Management** – 7,020 real products from a US county's liquor authority. Stock alerts, demand forecast, ABC analysis and a restock list. It flagged $2.5M of monthly sales at risk from stock-outs.

**4. Loan Approval Prediction** – 174,000 real US mortgage applications. XGBoost reaches ROC-AUC 0.90, explains every prediction, and is audited for fairness.

What I learned: check every model against a simple baseline, never let test data leak into training, and read the data before trusting it (stock counted in cans vs cases changed my inventory value by 2×).

Tech: Python, Pandas, Scikit-learn, XGBoost, PyTorch, Plotly, Streamlit.

Live demo: [link] · Code: [GitHub link]

#DataScience #MachineLearning #Python #Streamlit #Internship

_Attach: the screenshots in `reports/screenshots/`._
