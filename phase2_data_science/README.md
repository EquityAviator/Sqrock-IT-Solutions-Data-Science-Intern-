# Sqrock IT Solutions · Data Science Internship · Phase 2

Four end-to-end projects on **real, public data** (about 19 MB in total), one Streamlit dashboard,
GPU-trained models, and a crash-proof run logger.

| # | Project | Real dataset | Best model | Headline result (unseen test data) |
|---|---|---|---|---|
| 1 | Stock Market Prediction System | Yahoo Finance, 10 US stocks + S&P 500, 2015 → Nov 2026 | LSTM (GPU) / XGBoost (GPU) | mean R² **0.988**, price error **1.47%**; matches the "tomorrow = today" baseline, direction right **53%** |
| 2 | AI Data Analysis Dashboard | any CSV / Excel / JSON upload (sample: UCI Bank Marketing, 41,188 rows) | statistical tests + Isolation Forest | ranked findings with recommendations, HTML report |
| 3 | Smart Inventory Management System | Montgomery County (USA) Alcohol Beverage Services: 7,020 products, 352k sales rows, 305 suppliers | Moving average (beats ML here) | forecast error **21.0%** WAPE; 1,962 stock-outs and $2.5M monthly sales at risk flagged |
| 4 | Loan Approval Prediction System | US government HMDA 2023, Colorado: **174,143** mortgage applications | XGBoost (GPU) | ROC-AUC **0.901**, accuracy **84.8%**, catches **77%** of denials |

## Run it

```bash
pip install -r requirements.txt
# GPU build of PyTorch for the LSTM (optional; everything else runs without it):
pip install torch --index-url https://download.pytorch.org/whl/cu124

python download_data.py                      # fetch all datasets again (optional, they are included)

python task1_stock_prediction/train.py       # ~1 min with a GPU
python task3_inventory/pipeline.py           # ~1 min
python task4_loan_approval/train.py          # ~2 min

streamlit run app.py                         # the dashboard, http://localhost:8501
```

Sign-in: the app opens on a login screen (create an account, or continue as guest). Passwords are stored
only as salted PBKDF2 hashes in `users.json`. Set `PHASE2_AUTH=off` to skip it.

```bash
python tests/test_features.py                # 16 functional checks (login, uploads, PDFs, live data, predictions)
python tools/screenshots.py                  # PNG of every page into reports/screenshots/
```

To record everything a run does (output, CPU, RAM, GPU, disk, network, Windows driver errors),
start it through the logger. Each line is forced to disk immediately, so the log survives a crash:

```bash
python tools/run_logged.py task4 python task4_loan_approval/train.py
python tools/run_logged.py app   python -m streamlit run app.py
# -> logs/<date>_<time>_<name>.log
```

## Layout

```
app.py                     Streamlit entry point (navigation for the 5 pages)
ui/                        one file per dashboard page + shared styling
download_data.py           reproducible downloader, writes data/SOURCES.json (provenance)
limits.py                  caps CPU threads (80%) and GPU memory (90%), lowers process priority
data/                      stocks/  inventory/  loans/  samples/
task1_stock_prediction/    features.py  train.py  predict.py  train.ipynb  artifacts/
task2_ai_dashboard/        engine.py  (cleaning, profiling, insights, report; no UI code)
task3_inventory/           pipeline.py  pipeline.ipynb  artifacts/
task4_loan_approval/       prep.py  train.py  train.ipynb  artifacts/
tools/                     run_logged.py (crash-proof logger), make_notebooks.py, screenshots.py
tests/                     test_features.py
ui/auth.py, ui/pdf.py      login system, PDF report builder
reports/                   Phase2_Report.md, linkedin_post.md
```

## Data sources (all real, nothing simulated)

| Task | Source | Link |
|---|---|---|
| 1 | Yahoo Finance daily OHLCV via `yfinance` | https://finance.yahoo.com |
| 2 | UCI ML Repository, Bank Marketing (Moro, Cortez & Rita, 2014) | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| 3 | Montgomery County, Maryland open data: store inventory, warehouse & retail sales, purchases | https://data.montgomerycountymd.gov |
| 4 | US Consumer Financial Protection Bureau, Home Mortgage Disclosure Act data | https://ffiec.cfpb.gov/data-browser/ |

Two inventory fields are not published by the source and are derived with a stated rule
(`task3_inventory/pipeline.py`): units per case (from bottle / pack size) and purchase cost
(shelf price less an assumed 28% margin).

## Bonus features included

| Task | Bonus features |
|---|---|
| 1 | LSTM deep-learning model, XGBoost, candlestick charts, advanced indicators (RSI, MACD, Bollinger), multiple-stock comparison, real-time data, buy/sell signal with backtest, **news sentiment analysis**, **portfolio performance analysis** |
| 2 | JSON upload, Isolation Forest anomaly detection, HTML and **PDF** report |
| 3 | Automated restocking suggestions, supplier module, cost estimate, Excel / CSV / **PDF** export, **QR code and barcode labels with scan lookup**, **sales channels (stores vs wholesale)** |
| 4 | XGBoost, multi-model comparison, risk score, explainable AI (SHAP), feature importance, eligibility checker, recommendations, **PDF assessment**, fairness audit |
| All | **User authentication**, dark mode (Streamlit menu: Settings > Theme), interactive dashboard |

Not built, because the real data cannot support it: multi-warehouse stock. The source publishes one stock
total per product, not stock per location; the Channels tab shows the real split it does publish
(own stores vs wholesale) instead of inventing warehouses.

## Method notes

- **No leakage.** Time-ordered splits for stocks and inventory; scalers and encoders fitted on training
  rows only; the model is chosen on a validation set and the test set is used once.
- **Honest baselines.** Every forecast is scored against a naive baseline. For stocks a high R² alone
  proves nothing, because tomorrow's price is nearly today's; the report says so.
- **GPU.** The LSTM (PyTorch, mixed precision) and XGBoost train on the NVIDIA card and fall back to CPU.
- **Fair lending.** Sex, age, race and ethnicity are not model inputs for Task 4; they are used only in a
  fairness audit of the finished model.

## Deploying (for the internship's live-link requirement)

Push this folder to GitHub, then on [Streamlit Community Cloud](https://streamlit.io/cloud) create an
app with main file `app.py`. The dashboard only reads the saved `artifacts/`, so no GPU is needed online.
