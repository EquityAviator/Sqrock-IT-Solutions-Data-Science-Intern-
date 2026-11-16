"""Landing page: what Phase 2 contains and the headline result of each project."""
from __future__ import annotations

import json

import streamlit as st

from common import COLORS, ROOT, card, page

page("Data Science Internship · Phase 2", "Sqrock IT Solutions · four end-to-end projects on real, public data")


def metrics(task: str) -> dict | None:
    p = ROOT / task / "artifacts" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


t1, t3, t4 = metrics("task1_stock_prediction"), metrics("task3_inventory"), metrics("task4_loan_approval")
src_file = ROOT / "data" / "SOURCES.json"
src = json.loads(src_file.read_text()) if src_file.exists() else {}

c = st.columns(4)
if t1:
    best = min((k for k in t1["summary"] if not k.startswith("Naive")), key=lambda k: t1["summary"][k]["rmse_vs_naive"])
    c[0].metric("Stocks · best model", best, f"mean R² {t1['summary'][best]['mean_R2']:.4f}", delta_color="off")
c[1].metric("Dashboard · file types", "CSV · Excel · JSON", "auto-insights", delta_color="off")
if t3:
    c[2].metric("Inventory · products tracked", f"{t3['kpi']['total_products']:,}", f"{t3['kpi']['out_of_stock']:,} out of stock", delta_color="inverse")
if t4:
    r = t4["models"][t4["best_model"]]["test"]
    c[3].metric("Loans · ROC-AUC", f"{r['ROC_AUC']:.3f}", t4["best_model"], delta_color="off")

st.markdown("### Projects")
a, b = st.columns(2)
with a:
    card("Task 1", COLORS["primary"], "Stock Market Prediction System",
         "Ten large US companies, daily prices from Yahoo Finance since 2015. Linear Regression, Decision Tree, Random Forest, "
         "XGBoost and an LSTM trained on the GPU, each scored against a naive baseline. Candlestick charts, indicators, "
         "30-day forecast, buy/sell signal with backtest, live prices.")
    st.page_link("ui/stock.py", label="Open Stock Market Prediction", icon=":material/candlestick_chart:")
    card("Task 3", COLORS["amber"], "Smart Inventory Management System",
         "Real stock-on-hand, monthly sales and supplier records of a public liquor retailer. Stock status and alerts, "
         "demand forecast per product, ABC analysis, supplier performance, restocking list, Excel export.")
    st.page_link("ui/inventory.py", label="Open Smart Inventory", icon=":material/inventory_2:")
with b:
    card("Task 2", COLORS["green"], "AI Data Analysis Dashboard",
         "Upload any CSV, Excel or JSON file. The dashboard validates and cleans it, profiles every column, draws the charts you pick, "
         "and writes ranked findings with recommendations using significance tests and an Isolation Forest. Report export.")
    st.page_link("ui/dashboard.py", label="Open AI Data Analysis", icon=":material/insights:")
    card("Task 4", COLORS["violet"], "Loan Approval Prediction System",
         "174,000 real mortgage applications (US government HMDA data). Five classifiers compared, eligibility checker with an "
         "explanation for every prediction, risk score, and a fairness audit.")
    st.page_link("ui/loan.py", label="Open Loan Approval", icon=":material/account_balance:")

st.markdown("### Data sources")
st.caption("Every dataset is real and public. Nothing is simulated. `python download_data.py` fetches all of it again (about 20 MB).")
rows = []
for key, s in src.items():
    n = s.get("rows")
    n = sum(n.values()) if isinstance(n, dict) else n
    rows.append({"Used in": key.split("_")[0].replace("task", "Task "), "Source": s["source"], "Contents": s["what"],
                 "Rows": f"{n:,}" if n else "", "Link": s["url"], "Downloaded (UTC)": s["downloaded_utc"]})
if rows:
    st.dataframe(rows, width="stretch", hide_index=True, column_config={"Link": st.column_config.LinkColumn(display_text="open")})

st.markdown("### How it was built")
st.markdown("""
- **Pipelines** in `task*/` are plain Python scripts. Each one cleans its data, trains and tests the models, and writes results to `artifacts/`.
- **This dashboard** only reads those results, so it starts fast and can be hosted for free.
- **GPU**: the LSTM (PyTorch, mixed precision) and XGBoost train on the NVIDIA card when one is present, and fall back to CPU otherwise.
- **No leakage**: time-ordered splits for time series, scalers fitted on training data only, model choice on a validation set, test set used once.
""")
