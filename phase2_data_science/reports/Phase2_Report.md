# Phase 2 Report · Data Science Internship · Sqrock IT Solutions

All four tasks are complete. Every dataset is real and public; all numbers below are measured on
data the models never saw during training.

## Task 1 — Stock Market Prediction System

**Data.** Daily Open / High / Low / Close / Volume from Yahoo Finance for Apple, Microsoft, Alphabet,
Amazon, NVIDIA, Tesla, Meta, JPMorgan, Johnson & Johnson and Exxon, plus the S&P 500, from
2 Jan 2015 to 13 Nov 2026 (2,985 trading days each).

**Preprocessing.** Dates parsed, duplicate dates removed, missing quotes forward-filled, non-positive
prices rejected, prices adjusted for splits and dividends. 30 features: returns over 1–20 days, gaps to
5/10/20/50/200-day moving averages, MACD, RSI, Bollinger bands, volatility, volume ratio, market return
and beta. Features standardised with a scaler fitted on training rows only.

**Models.** Linear Regression, Decision Tree, Random Forest, XGBoost (GPU) per stock; one LSTM
(2 layers, 60-day windows, PyTorch on the GTX 1080 with mixed precision) across all ten stocks.
Each predicts tomorrow's return, converted to a price. Train < 2023, validate 2023, test 2024 → Nov 2026.

| Model | Mean R² | Mean MAPE | RMSE vs naive | Direction accuracy | Stocks beating naive |
|---|---|---|---|---|---|
| Naive (tomorrow = today) | 0.98816 | 1.479% | 1.0000 | – | – |
| Linear Regression | 0.98768 | 1.508% | 1.0177 | 50.0% | 1 / 10 |
| Decision Tree | 0.98789 | 1.493% | 1.0127 | 51.6% | 1 / 10 |
| Random Forest | 0.98803 | 1.482% | 1.0034 | 51.1% | 2 / 10 |
| XGBoost (GPU) | 0.98815 | 1.475% | 1.0002 | 52.5% | 5 / 10 |
| **LSTM (GPU)** | **0.98816** | **1.473%** | **0.9996** | **53.2%** | 4 / 10 |

**Interpretation.** Prices are predicted to within about 1.5%, with R² 0.988. But the naive guess scores
the same, so the models add almost nothing on price level; their edge is a small one on direction (53%).
That is the expected result for liquid large-cap stocks and is reported as such. The buy/sell signal
backtest confirms it: on no stock did trading the signal beat simply holding.

**Delivered.** Candlestick charts with moving averages, Bollinger bands, RSI, MACD and volume;
multi-stock comparison and correlation heatmap; actual vs predicted; 30-day forecast with an 80% range;
buy/sell signal with backtest; live prices from Yahoo Finance.

## Task 2 — AI Data Analysis Dashboard

Upload CSV, Excel or JSON (validated for type, size, emptiness and parse errors). The engine
(`task2_ai_dashboard/engine.py`) then:

- **Cleans** the data and logs every operation: header tidy-up, empty rows/columns, text → number / date
  conversion, standardised missing markers, duplicate removal, median / most-frequent imputation, optional scaling.
- **Profiles** each column as numeric, categorical, date, identifier or text.
- **Finds insights** automatically and ranks them: outcome drivers (chi-square with Cramér's V, Cohen's d),
  trends and seasonality (linear trend test), concentration and top performers (Pareto, ANOVA), strong
  correlations, anomalies (3×IQR rule and an Isolation Forest), and data-quality warnings. Each finding
  carries a plain-language recommendation.
- **Reports**: self-contained HTML report with overview, cleaning log, findings and charts.

On the UCI Bank Marketing sample (41,188 rows) it correctly identifies that 11.3% of contacts subscribed,
that a previous successful campaign lifts the rate to 65.1% (vs 8.8%), and that March (50.5%) far
outperforms May (6.4%).

Dashboard: sidebar dataset controls, overview, preprocessing log, seven chart types with dynamic column
selection, insights, analytics cards, anomalies, search / filter / custom conditions, report export.

## Task 3 — Smart Inventory Management System

**Data.** Montgomery County (Maryland) Alcohol Beverage Services open data: current stock and shelf
price of 7,020 products, 352,419 monthly sales rows by item and supplier (28 months between 2017 and
2026), and 8,649 purchase-order lines. 102 categories, 305 suppliers.

**Preprocessing.** Types fixed; 217 missing suppliers labelled; 121 negative adjustment rows excluded
from demand; 737 non-merchandise rows removed. Sales cases converted to stock units using pack size
(verified against the data: multipacks are counted in single cans, loose beer cases in cases).

**Current position.** Stock 4.68M units worth **$35.6M** at retail. Status: 1,757 healthy, 475 low,
278 critical, **1,962 out of stock** (with live demand), 257 overstocked ($5.0M tied up), 2,291 inactive.
Sales at risk next month from stock-outs: **$2.5M**. Restock list: 2,715 products, 155,509 cases, about $9.6M.

**Demand forecast** (tested on Aug and Sep 2026):

| Model | MAE (cases) | WAPE | Bias |
|---|---|---|---|
| Naive (last month) | 8.04 | 22.2% | +3.0% |
| **Moving average (recent months)** | **7.62** | **21.0%** | +10.3% |
| Linear Regression | 10.69 | 29.5% | +5.5% |
| Random Forest | 7.68 | 21.2% | +3.0% |
| XGBoost (GPU) | 8.22 | 22.7% | +6.4% |

The simple moving average wins. The published history has a six-year gap, so the machine-learning
models train mostly on 2017–2020 and cannot learn much that the recent average does not already hold.
The system uses whichever model tests best.

**Insights.** ABC analysis: 1,361 products (21%) bring 80% of revenue. Fast / slow movers, overstock,
supplier performance (revenue, stock-out rate, delivered vs sold) and category performance.

## Task 4 — Loan Approval Prediction System

**Data.** Every mortgage application decided in Colorado in 2023 (HMDA, US Consumer Financial
Protection Bureau): 183,015 rows, **174,143** after cleaning; 78.1% approved.

**Preprocessing.** 245 duplicates removed; business loans, multifamily and partially-exempt reporters
excluded; impossible incomes and ratios set to missing; extreme values capped at the 99.9th percentile;
median imputation with missing-value flags; scaling; one-hot encoding. Engineered: loan-to-income,
payment-to-income, loan-to-value. Split 70 / 15 / 15, stratified.

| Model | Accuracy | Precision | Recall | F1 | Denials caught | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.749 | 0.909 | 0.753 | 0.824 | 0.732 | 0.830 |
| Decision Tree | 0.803 | 0.923 | 0.816 | 0.866 | 0.759 | 0.871 |
| Random Forest | 0.818 | 0.936 | 0.823 | 0.876 | 0.800 | 0.896 |
| SVM (linear) | 0.730 | 0.913 | 0.723 | 0.807 | 0.755 | 0.830 |
| **XGBoost (GPU)** | **0.848** | 0.932 | **0.869** | **0.899** | 0.774 | **0.901** |

Thresholds are tuned for balanced accuracy, because always answering "approved" already scores 78.1%.
At the plain 50% cut-off XGBoost reaches 89.0% accuracy but catches only 59% of denials.
Confusion matrix (XGBoost, 26,122 test applications): 17,718 approvals and 4,434 denials correct;
1,298 denials missed; 2,672 good applications flagged for review.

**What drives decisions.** Debt-to-income 42% of predictive power, loan purpose 21%, loan-to-value 11%.
Above 60% debt-to-income only 12% are approved, against 89% at 30–36%. Lenders' own top stated denial
reasons: debt-to-income (37%), credit history (23%), collateral (13%).

**Fairness audit.** Sex, age, race and ethnicity are not model inputs. The model's approval rates still
differ between groups (for example Hispanic or Latino 61% vs not Hispanic 75%), mirroring gaps in the
lenders' real decisions (66% vs 80%). Reported as a reason for human review, not as proof of bias.

**Delivered.** Eligibility checker with calibrated probability, risk score, SHAP explanation of each
prediction and tailored advice; approval statistics; applicant demographics; model comparison, ROC,
confusion matrix, threshold and calibration charts; insights.

## Engineering

- One Streamlit app (`app.py`) with sidebar navigation; pages read saved artifacts and load in seconds.
- `limits.py` keeps CPU use to 80% of threads and GPU memory to 90%.
- `tools/run_logged.py` records program output and system state every 3 seconds, fsync'ed to disk.
- Executed Jupyter notebooks for Tasks 1, 3 and 4 are generated from the scripts.

## Bonus features

- **Task 1:** news sentiment (live Yahoo Finance headlines scored with VADER) and portfolio analysis
  (return, yearly return, volatility, Sharpe ratio, drawdown, beta and contribution per holding vs the S&P 500).
- **Task 3:** QR code and Code 128 barcode for every product, lookup by scanned code, printable label sheet;
  sales channels (own stores vs wholesale) with a store replenishment-gap list.
- **PDF export:** analysis report (Task 2), inventory report (Task 3), loan assessment (Task 4).
- **User authentication:** register / sign in / guest, PBKDF2-hashed passwords, attempt limit.
- **Testing:** `tests/test_features.py` runs 16 functional checks; all pass. Screenshots of every page are in
  `reports/screenshots/`.

## Known limits

- Multi-warehouse management is not built: the source publishes one stock total per product, not per location.

- Stock models do not beat the naive baseline on price; they should not be used as trading advice.
- Inventory sales history has gaps (2018 → 2026), which limits forecasting accuracy; purchase cost is estimated.
- HMDA has no credit score, education or employment field, so those brief items are covered by
  debt-to-income, loan-to-value and the lender's stated denial reasons instead.
